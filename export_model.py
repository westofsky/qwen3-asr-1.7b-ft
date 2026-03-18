#!/usr/bin/env python3
"""
학습된 체크포인트를 다른 환경에서 바로 쓸 수 있는 완전한 모델 디렉터리로 저장.

Usage:
  python export_model.py --checkpoint ./checkpoints --output ./exported_model
  python export_model.py --checkpoint ./checkpoints/checkpoint-1000 --output ./exported_model
"""
import json, argparse, os, re, shutil
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

def find_best_checkpoint(checkpoint_dir: str) -> str:
    p = Path(checkpoint_dir)

    # 직접 체크포인트 경로
    if (p / "config.json").exists():
        return str(p)

    # trainer_state.json에서 best 찾기
    state_file = p / "trainer_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            print(f"Best checkpoint: {best}")
            return best

    # 가장 최신 checkpoint
    checkpoints = []
    for name in os.listdir(p):
        m = _CKPT_RE.match(name)
        if m and os.path.isdir(p / name):
            checkpoints.append((int(m.group(1)), str(p / name)))
    if checkpoints:
        return sorted(checkpoints)[-1][1]

    raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints")
    parser.add_argument("--output",     default="./exported_model")
    args = parser.parse_args()

    ckpt = find_best_checkpoint(args.checkpoint)
    print(f"Loading: {ckpt}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        ckpt,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    asr_wrapper.model.save_pretrained(out)
    asr_wrapper.processor.save_pretrained(out)

    print(f"\nExported to: {out}")
    print("\n--- 추론 예시 ---")
    print("from qwen_asr import Qwen3ASRModel")
    print(f"model = Qwen3ASRModel.from_pretrained('{out}')")
    print("result = model.transcribe('test.wav')")
    print("print(result)")


if __name__ == "__main__":
    main()
