#!/usr/bin/env python3
"""
오디오 추출 없이 zip에서 직접 manifest만 생성.
manifest 포맷: {"zip_path", "audio_key", "text", "prompt", "duration", "format", "source"}

Main : 007.저음질_전화망_음성인식_데이터 (WAV 8kHz, phone narrowband)
Aux  : 159.숫자가_포함된_패턴_발화_데이터 (PCM 16kHz clean)

007 텍스트 정제 규칙:
  1) (A)/(B) → A          # 앞 괄호 내용을 정답으로
  2) o/ n/ b/ l/ 제거      # 전사 보조 마커 제거
  3) (()) 제거             # 불명료 구간 마커 제거
  4) 연속 공백 정리

159 텍스트 규칙:
  scriptITN 사용 (숫자가 아라비아 숫자로 표기된 버전)

Usage:
  python prepare_dataset.py
  python prepare_dataset.py --max_samples 50000
"""
import re, json, zipfile, struct, argparse, random
from pathlib import Path
from collections import defaultdict

import yaml
from tqdm import tqdm


def load_config(path: str = "config.yaml") -> dict:
    """config.yaml을 argparse 기본값 소스로 사용 (CLI가 항상 우선)."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


# ── helpers ───────────────────────────────────────────────────────────────────

def decode_zip_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        return name


def clean_main_transcript(text: str) -> str:
    """007 전화망 데이터의 전사 마커 제거."""
    if not text:
        return ""
    # 1) (A)/(B) → A  (앞 괄호 내용 채택)
    text = re.sub(r"\(([^)]*)\)/\(([^)]*)\)", r"\1", text)
    # 2) 단일 문자 슬래시 마커 제거: o/ n/ b/ l/ (AI-Hub 007 전사 규약)
    text = re.sub(r"\b[onbl]/\s*", "", text)
    # 3) 불명료 구간 마커 (()) 제거
    text = re.sub(r"\(\(\)\)", "", text)
    # 4) 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


def probe_wav_format(z: zipfile.ZipFile, key: str):
    try:
        with z.open(key) as f:
            header = f.read(44)
        if len(header) < 44 or header[:4] != b"RIFF":
            return None
        channels    = struct.unpack_from("<H", header, 22)[0]
        sample_rate = struct.unpack_from("<I", header, 24)[0]
        bit_depth   = struct.unpack_from("<H", header, 34)[0]
        return sample_rate, channels, bit_depth
    except Exception:
        return None


def wav_duration_from_size(file_size, sr, ch, bd):
    return (file_size - 44) / (sr * ch * (bd // 8))


def pcm_duration_from_size(file_size, sr=16000, bd=16):
    return file_size / (sr * (bd // 8))


def find_subdirs_containing(base: Path, keyword: str):
    if not base.exists():
        return []
    return [d for d in base.iterdir() if d.is_dir() and keyword in d.name]


# ── main dataset (007.저음질_전화망_음성인식_데이터) ────────────────────────

def process_main(main_dir: str, split: str, min_dur: float, max_dur: float, max_samples: int) -> list:
    split_path = "1.Training" if split == "train" else "2.Validation"
    base = Path(main_dir) / "01.데이터" / split_path

    label_zips = sorted(p for d in find_subdirs_containing(base, "라벨링데이터") for p in d.glob("*.zip"))
    wav_zips   = sorted(p for d in find_subdirs_containing(base, "원천데이터")   for p in d.glob("*.zip"))

    print(f"\n[main/{split}] label zips={len(label_zips)}, wav zips={len(wav_zips)}")
    if not label_zips or not wav_zips:
        return []

    # wav index: internal_key → (abs_zip_path, file_size)
    wav_index: dict = {}
    for zp in tqdm(wav_zips, desc="  Indexing wav zips"):
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.filename.endswith(".wav"):
                    wav_index[info.filename] = (str(zp.resolve()), info.file_size)
    print(f"  WAV index: {len(wav_index):,}")

    # 첫 WAV로 포맷 probe (007은 8kHz/1ch/16bit 예상)
    wav_fmt = None
    if wav_index:
        k, (zp, _) = next(iter(wav_index.items()))
        with zipfile.ZipFile(zp) as z:
            wav_fmt = probe_wav_format(z, k)
        if wav_fmt:
            print(f"  WAV format: {wav_fmt[0]}Hz {wav_fmt[1]}ch {wav_fmt[2]}bit")

    # label zip의 JSON dialogs에서 (audioPath, text) 직접 추출
    pairs: list = []
    for lz_path in tqdm(label_zips, desc="  Parsing label zips"):
        with zipfile.ZipFile(lz_path) as lz:
            json_list = [n for n in lz.namelist() if n.endswith(".json")]
            for jf in tqdm(json_list, desc=f"    {lz_path.name}", leave=False, mininterval=2):
                try:
                    data = json.loads(lz.read(jf))
                    for dlg in data["dataSet"]["dialogs"]:
                        ap = dlg.get("audioPath", "")
                        transcript = clean_main_transcript(dlg.get("text", ""))
                        if transcript and ap in wav_index:
                            pairs.append((ap, transcript))
                except Exception:
                    pass

    print(f"  Pairs found: {len(pairs):,}")
    if max_samples and len(pairs) > max_samples:
        random.shuffle(pairs)
        pairs = pairs[:max_samples]
        print(f"  Subsampled: {len(pairs):,}")

    records = []
    for ap, transcript in pairs:
        zip_path, file_size = wav_index[ap]
        duration = wav_duration_from_size(file_size, *wav_fmt) if wav_fmt else None
        if duration is not None and (duration < min_dur or duration > max_dur):
            continue
        records.append({
            "zip_path":  zip_path,
            "audio_key": ap,
            "text":      transcript,
            "prompt":    "",
            "duration":  round(duration, 3) if duration else None,
            "format":    "wav",
            "source":    "main",
        })

    print(f"  Records after filter: {len(records):,}")
    return records


# ── aux dataset (159.숫자가_포함된_패턴_발화_데이터) ─────────────────────────

def process_aux(aux_dir: str, split: str, min_dur: float, max_dur: float) -> list:
    split_path = "1.Training" if split == "train" else "2.Validation"
    src_dir = Path(aux_dir) / "01.데이터" / split_path / "원천데이터"

    txt_zips   = sorted(src_dir.glob("*텍스트*.zip"))
    audio_zips = sorted(src_dir.glob("*음성*.zip"))

    if not txt_zips or not audio_zips:
        print(f"\n[aux/{split}] zip 없음: {src_dir}")
        return []

    txt_zip, audio_zip = txt_zips[0], audio_zips[0]
    print(f"\n[aux/{split}] text={txt_zip.name}, audio={audio_zip.name}")

    # scriptID → scriptITN 매핑 (사용자 요청: 무조건 scriptITN)
    script_map: dict = {}
    with zipfile.ZipFile(txt_zip) as tz:
        for name in tqdm(tz.namelist(), desc="  Parsing text zip"):
            if not name.endswith(".txt"):
                continue
            content = tz.read(name).decode("utf-8", errors="ignore")
            sid, itn = None, None
            for line in content.splitlines():
                if line.startswith("scriptID"):
                    sid = line.split(":", 1)[1].strip()
                elif line.startswith("scriptITN"):
                    itn = line.split(":", 1)[1].strip()
            if sid and itn:
                script_map[sid] = itn  # ITN은 별도 정제 불필요
    print(f"  Scripts: {len(script_map):,}")

    # audio index: scriptID(디렉터리명) → [(internal_key, file_size), ...]
    audio_index: dict = defaultdict(list)
    with zipfile.ZipFile(audio_zip) as az:
        for info in az.infolist():
            if not info.filename.endswith(".pcm"):
                continue
            decoded = decode_zip_name(info.filename)
            parts = decoded.split("/")
            if len(parts) >= 3:
                audio_index[parts[1]].append((info.filename, info.file_size))
    print(f"  Audio script IDs: {len(audio_index):,}")

    audio_zip_abs = str(audio_zip.resolve())
    records = []
    for sid, items in tqdm(audio_index.items(), desc="  Building aux manifest"):
        transcript = script_map.get(sid, "")
        if not transcript:
            continue
        for audio_key, file_size in items:
            if file_size == 0:
                continue
            duration = pcm_duration_from_size(file_size)
            if duration < min_dur or duration > max_dur:
                continue
            records.append({
                "zip_path":  audio_zip_abs,
                "audio_key": audio_key,
                "text":      transcript,
                "prompt":    "",
                "duration":  round(duration, 3),
                "format":    "pcm",
                "source":    "aux",
            })

    print(f"  Records: {len(records):,}")
    return records


# ── mix & write ───────────────────────────────────────────────────────────────

def mix(main_records, aux_records, aux_ratio, seed):
    random.seed(seed)
    if not main_records:
        return list(aux_records)
    n_aux_target = int(len(main_records) * aux_ratio / max(1 - aux_ratio, 1e-6))
    aux_sample = random.sample(aux_records, min(n_aux_target, len(aux_records)))
    if len(aux_records) < n_aux_target:
        print(f"  [warn] aux {len(aux_records):,} < target {n_aux_target:,}")
    combined = main_records + aux_sample
    random.shuffle(combined)
    ratio = len(aux_sample) / max(len(combined), 1) * 100
    print(f"  main={len(main_records):,}  aux={len(aux_sample):,}  total={len(combined):,}  aux%={ratio:.1f}%")
    return combined


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote: {path}  ({len(records):,} records)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir",     default=cfg.get("main_dir",  "/mnt/data/007.저음질_전화망_음성인식_데이터"))
    parser.add_argument("--aux_dir",      default=cfg.get("aux_dir",   "/mnt/data/159.숫자가_포함된_패턴_발화_데이터"))
    parser.add_argument("--output_dir",   default=cfg.get("data_dir",  "./data"))
    parser.add_argument("--aux_ratio",    type=float, default=cfg.get("aux_ratio",    0.15))
    parser.add_argument("--max_duration", type=float, default=cfg.get("max_duration", 30.0))
    parser.add_argument("--min_duration", type=float, default=cfg.get("min_duration", 0.3))
    parser.add_argument("--max_samples",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=cfg.get("seed", 42))
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_train = process_main(args.main_dir, "train", args.min_duration, args.max_duration, args.max_samples)
    main_valid = process_main(args.main_dir, "valid", args.min_duration, args.max_duration, 0)
    aux_train  = process_aux(args.aux_dir,  "train", args.min_duration, args.max_duration)
    aux_valid  = process_aux(args.aux_dir,  "valid", args.min_duration, args.max_duration)

    print("\n--- Mixing train ---")
    train_records = mix(main_train, aux_train, args.aux_ratio, args.seed)
    print("--- Mixing valid ---")
    valid_records = mix(main_valid, aux_valid, args.aux_ratio, args.seed)

    write_jsonl(train_records, out / "train.jsonl")
    write_jsonl(valid_records, out / "valid.jsonl")
    print("\nDone.")


if __name__ == "__main__":
    main()
