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
import soundfile as sf
from datasets import Dataset as HFDataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


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
    _prefix_len: int = -1
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

        if self._prefix_len < 0:
            prefix_inputs = self.processor(
                text=[prefix_texts[0]], audio=[audios[0]],
                return_tensors="pt", truncation=False,
            )
            self._prefix_len = int(prefix_inputs["attention_mask"].sum().item())

        labels = full_inputs["input_ids"].clone()
        labels[:, :self._prefix_len] = -100

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

def build_hf_dataset(manifest_path: str, processor) -> HFDataset:
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",      default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--train_file",      default="./data/train.jsonl")
    parser.add_argument("--eval_file",       default="./data/valid.jsonl")
    parser.add_argument("--output_dir",      default="./checkpoints")
    parser.add_argument("--sr",              type=int,   default=16000)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--grad_acc",        type=int,   default=2)     # effective batch=64
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--max_steps",       type=int,   default=5000)
    parser.add_argument("--save_steps",      type=int,   default=500)
    parser.add_argument("--log_steps",       type=int,   default=25)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--aux_phone_prob",  type=float, default=1.0,
                        help="aux(159) clean 음성에 4kHz 대역제한 적용 확률")
    parser.add_argument("--main_phone_prob", type=float, default=0.0,
                        help="main(007) 음성에 추가 대역제한 적용 확률 (기본 off: 이미 8k 소스)")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--resume",          action="store_true")
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
    train_ds = build_hf_dataset(args.train_file, processor)
    eval_ds  = build_hf_dataset(args.eval_file, processor) if args.eval_file else None
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
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeCheckpointInferableCallback(args.model_path)],
    )

    resume_from = find_latest_checkpoint(args.output_dir) if args.resume else None
    if resume_from:
        print(f"Resuming from: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)
    print(f"\nDone. Checkpoints: {args.output_dir}")


if __name__ == "__main__":
    main()
