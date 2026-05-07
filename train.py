#!/usr/bin/env python3
"""
Qwen3-ASR-1.7B full fine-tuning
- 공식 qwen-asr 패키지 사용 (Qwen3ASRModel)
- zip에서 on-the-fly로 오디오 읽기 (추출 없음)
- bf16, gradient checkpointing
- Asterisk 16k (narrowband-in-16k) 타깃:
    aux(159, 16k clean) 샘플에 4kHz band-limit 필터를 기본 적용,
    main(007, 8k 전화망) 샘플은 이미 narrowband이므로 업샘플만.
"""
import argparse, io, json, os, random, re, shutil, zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
import yaml
import soundfile as sf
from datasets import Dataset as HFDataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


def load_config(path: str = "config.yaml") -> dict:
    """config.yaml을 argparse 기본값 소스로 사용 (CLI가 항상 우선)."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


# ── zip 캐시 (프로세스별 전역) ────────────────────────────────────────────────

_zip_cache: dict = {}

def _read_from_zip(zip_path: str, key: str) -> bytes:
    global _zip_cache
    if zip_path not in _zip_cache:
        _zip_cache[zip_path] = zipfile.ZipFile(zip_path, "r")
    return _zip_cache[zip_path].read(key)


def load_audio_from_record(rec: dict, sr: int = 16000) -> np.ndarray:
    """zip에서 오디오를 읽어 float32 numpy 배열로 반환 (sr로 리샘플)."""
    raw = _read_from_zip(rec["zip_path"], rec["audio_key"])

    if rec["format"] == "pcm":
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        orig_sr = 16000
    else:
        audio, orig_sr = sf.read(io.BytesIO(raw), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr, res_type="kaiser_fast")

    return audio


# ── Asterisk narrowband-in-16k 시뮬레이션 ─────────────────────────────────────

def phone_band_filter(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """16k→8k→16k 라운드트립으로 4kHz 대역제한.
    Asterisk가 narrowband(PCMU/PCMA/G.729) 통화를 16k로 올려 전달하는 신호를 근사."""
    if sr != 16000 or len(audio) < 32:
        return audio
    lo = librosa.resample(audio, orig_sr=sr, target_sr=8000, res_type="kaiser_fast")
    up = librosa.resample(lo, orig_sr=8000, target_sr=sr, res_type="kaiser_fast")
    # 길이 보정 (resample이 미세하게 길이 바꿀 수 있음)
    if len(up) > len(audio):
        up = up[:len(audio)]
    elif len(up) < len(audio):
        up = np.pad(up, (0, len(audio) - len(up)))
    return up.astype(np.float32, copy=False)


# ── 공식 스크립트에서 가져온 유틸 ────────────────────────────────────────────

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                 feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# ── 평가 메트릭 (CER) ─────────────────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """문자 단위 Levenshtein distance (CER용)."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,        # insertion
                curr[j] + 1,            # deletion
                prev[j] + (c1 != c2),   # substitution
            ))
        prev = curr
    return prev[-1]


def preprocess_logits_for_metrics(logits, labels):
    """eval 시 누적 텐서를 argmax로 줄여 메모리 폭주 방지 (vocab=152K → 1)."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def make_compute_metrics(processor):
    """teacher-forced argmax 기준 CER (best checkpoint 선별용 일관 지표)."""
    def compute_metrics(eval_pred):
        preds, labels = eval_pred  # preds: argmax ids (N, T), labels: (N, T)
        # Causal LM shift: pred[t]가 label[t+1]을 예측
        preds = preds[..., :-1]
        labels = labels[..., 1:]

        total_chars, total_errors = 0, 0
        for p, l in zip(preds, labels):
            mask = l != -100
            if not mask.any():
                continue
            pred_str  = processor.tokenizer.decode(p[mask], skip_special_tokens=True)
            label_str = processor.tokenizer.decode(l[mask], skip_special_tokens=True)
            total_chars  += len(label_str)
            total_errors += _levenshtein(pred_str, label_str)
        return {"cer": total_errors / max(total_chars, 1)}

    return compute_metrics


# ── 체크포인트 유틸 ───────────────────────────────────────────────────────────

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def copy_required_hf_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for fn in ["config.json", "generation_config.json", "preprocessor_config.json",
               "processor_config.json", "tokenizer_config.json", "tokenizer.json",
               "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json"]:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


# ── 데이터 콜레이터 (zip에서 직접 읽기 + 전화망 시뮬레이션) ──────────────────

@dataclass
class DataCollatorZipASR:
    processor: Any
    sampling_rate: int = 16000
    aux_phone_prob: float = 1.0     # 159(clean 16k) → phone-band 강제 (Asterisk 매칭)
    main_phone_prob: float = 0.0    # 007은 이미 8k 전화망 소스라 불필요
    seed: int = 42
    _rng: Optional[random.Random] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # worker별 재현 가능한 RNG (seed + pid)
        self._rng = random.Random(self.seed + os.getpid())

    def _maybe_augment(self, audio: np.ndarray, source: str) -> np.ndarray:
        prob = self.aux_phone_prob if source == "aux" else self.main_phone_prob
        if prob > 0 and self._rng.random() < prob:
            audio = phone_band_filter(audio, self.sampling_rate)
        return audio

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = []
        for f in features:
            a = load_audio_from_record(f, sr=self.sampling_rate)
            a = self._maybe_augment(a, f.get("source", "main"))
            audios.append(a)

        prefix_texts = [f["prefix_text"] for f in features]
        targets      = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        full_inputs = self.processor(
            text=full_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )

        # Per-sample prefix length: 오디오 길이에 따라 audio token 수가 달라질 수 있으므로
        # 첫 샘플 값을 캐시하지 않고 매 샘플마다 계산.
        prefix_lens = []
        for pfx, aud in zip(prefix_texts, audios):
            pfx_inputs = self.processor(
                text=[pfx], audio=[aud],
                return_tensors="pt", truncation=False,
            )
            prefix_lens.append(int(pfx_inputs["attention_mask"].sum().item()))

        labels = full_inputs["input_ids"].clone()
        for i, plen in enumerate(prefix_lens):
            labels[i, :plen] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


class MakeCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        copy_required_hf_files(self.base_model_path, ckpt_dir)
        return control


# ── manifest 로드 + prefix_text 생성 ─────────────────────────────────────────

def build_hf_dataset(manifest_path: str, processor, max_label_length: Optional[int] = None) -> HFDataset:
    records = []
    skipped = 0
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            if max_label_length is not None and max_label_length > 0:
                tids = processor.tokenizer(rec["text"], add_special_tokens=False)["input_ids"]
                if len(tids) > max_label_length:
                    skipped += 1
                    continue
            records.append(rec)

    if skipped:
        print(f"  Skipped {skipped:,} records with > {max_label_length} target tokens")

    prefix_cache: dict = {}

    def get_prefix_text(prompt: str) -> str:
        if prompt not in prefix_cache:
            msgs = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "audio", "audio": None}]},
            ]
            prefix_cache[prompt] = processor.apply_chat_template(
                [msgs], add_generation_prompt=True, tokenize=False
            )[0]
        return prefix_cache[prompt]

    for rec in records:
        rec["target"]      = rec["text"]
        rec["prefix_text"] = get_prefix_text(rec.get("prompt", ""))

    return HFDataset.from_list(records)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    data_dir = cfg.get("data_dir", "./data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",        default=cfg.get("model_name_or_path", "Qwen/Qwen3-ASR-1.7B"))
    parser.add_argument("--train_file",        default=str(Path(data_dir) / "train.jsonl"))
    parser.add_argument("--eval_file",         default=str(Path(data_dir) / "valid.jsonl"))
    parser.add_argument("--output_dir",        default=cfg.get("output_dir", "./checkpoints"))
    parser.add_argument("--sr",                type=int,   default=16000)
    parser.add_argument("--batch_size",        type=int,   default=cfg.get("per_device_train_batch_size", 32))
    parser.add_argument("--grad_acc",          type=int,   default=cfg.get("gradient_accumulation_steps", 2))
    parser.add_argument("--lr",                type=float, default=cfg.get("learning_rate", 2e-5))
    parser.add_argument("--max_steps",         type=int,   default=cfg.get("max_steps", 5000))
    parser.add_argument("--save_steps",        type=int,   default=cfg.get("save_steps", 500))
    parser.add_argument("--log_steps",         type=int,   default=cfg.get("logging_steps", 25))
    parser.add_argument("--num_workers",       type=int,   default=cfg.get("dataloader_num_workers", 4))
    parser.add_argument("--max_label_length", type=int,   default=cfg.get("max_label_length", 448),
                        help="target 토큰이 이 길이 초과면 학습/평가에서 제외")
    parser.add_argument("--aux_phone_prob",    type=float, default=cfg.get("aux_phone_prob", 1.0),
                        help="aux(159) clean 음성에 4kHz 대역제한 적용 확률")
    parser.add_argument("--main_phone_prob",   type=float, default=cfg.get("main_phone_prob", 0.0),
                        help="main(007) 음성에 추가 대역제한 적용 확률 (기본 off: 이미 8k 소스)")
    parser.add_argument("--seed",              type=int,   default=cfg.get("seed", 42))
    parser.add_argument("--resume",            action="store_true")
    args = parser.parse_args()

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    print(f"Loading: {args.model_path}  bf16={use_bf16}")
    print(f"Phone-band aug: aux={args.aux_phone_prob}  main={args.main_phone_prob}")

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model     = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    print("Building datasets...")
    train_ds = build_hf_dataset(args.train_file, processor, args.max_label_length)
    eval_ds  = build_hf_dataset(args.eval_file,  processor, args.max_label_length) if args.eval_file else None
    print(f"Train: {len(train_ds):,}  Valid: {len(eval_ds) if eval_ds else 0:,}")

    collator = DataCollatorZipASR(
        processor=processor,
        sampling_rate=args.sr,
        aux_phone_prob=args.aux_phone_prob,
        main_phone_prob=args.main_phone_prob,
        seed=args.seed,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=args.log_steps,
        lr_scheduler_type="linear",
        warmup_ratio=0.02,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=(args.num_workers > 0),
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="cer" if eval_ds is not None else None,
        greater_is_better=False,
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=make_compute_metrics(processor) if eval_ds else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_ds else None,
        callbacks=[MakeCheckpointInferableCallback(args.model_path)],
    )

    resume_from = find_latest_checkpoint(args.output_dir) if args.resume else None
    if resume_from:
        print(f"Resuming from: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)
    print(f"\nDone. Checkpoints: {args.output_dir}")


if __name__ == "__main__":
    main()
