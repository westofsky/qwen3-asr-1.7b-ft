#!/usr/bin/env python3
"""
오디오 추출 없이 zip에서 직접 manifest만 생성.
manifest 포맷: {"zip_path", "audio_key", "text", "prompt", "duration", "format", "source"}

Usage:
  python prepare_dataset.py
  python prepare_dataset.py --max_samples 50000
"""
import re, json, zipfile, struct, argparse, random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def decode_zip_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        return name


def clean_script_tn(text: str) -> str:
    return re.sub(r"\[([^\]]*)\]", r"\1", text).strip()


def clean_main_transcript(text: str) -> str:
    # Remove leading "n/" prefix (e.g. "n/ 아/ 그러면..." → "아/ 그러면...")
    text = re.sub(r"^n/\s*", "", text)
    # Replace (A)/(B) with A (e.g. "(56)/(오 육)" → "56", "(렌탈)/(렌털)" → "렌탈")
    text = re.sub(r"\(([^)]+)\)/\([^)]+\)", r"\1", text)
    return text.strip()


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
    return [d for d in base.iterdir() if d.is_dir() and keyword in d.name]


# ── main dataset (012) ────────────────────────────────────────────────────────

def process_main(main_dir: str, split: str, min_dur: float, max_dur: float, max_samples: int) -> list:
    split_path = "1.Training" if split == "train" else "2.Validation"
    base = Path(main_dir) / "01.데이터" / split_path

    label_zips = sorted(p for d in find_subdirs_containing(base, "라벨링데이터") for p in d.glob("*.zip"))
    wav_zips   = sorted(p for d in find_subdirs_containing(base, "원천데이터")   for p in d.glob("*.zip"))

    print(f"\n[main/{split}] label zips={len(label_zips)}, wav zips={len(wav_zips)}")

    # wav index: internal_key → (abs_zip_path, file_size)
    wav_index: dict = {}
    for zp in tqdm(wav_zips, desc="  Indexing wav zips"):
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.filename.endswith(".wav"):
                    wav_index[info.filename] = (str(zp.resolve()), info.file_size)
    print(f"  WAV index: {len(wav_index):,}")

    # WAV 포맷 probe (첫 파일로 sample_rate 파악)
    wav_fmt = None
    if wav_index:
        k, (zp, _) = next(iter(wav_index.items()))
        with zipfile.ZipFile(zp) as z:
            wav_fmt = probe_wav_format(z, k)
        if wav_fmt:
            print(f"  WAV format: {wav_fmt[0]}Hz {wav_fmt[1]}ch {wav_fmt[2]}bit")

    # label zips 파싱 → (audio_key, transcript)
    pairs: list = []
    for lz_path in tqdm(label_zips, desc="  Parsing label zips"):
        with zipfile.ZipFile(lz_path) as lz:
            txt_map: dict = {}
            json_list: list = []
            for name in lz.namelist():
                if name.endswith(".txt"):
                    try:
                        txt_map[name] = lz.read(name).decode("utf-8", errors="ignore").strip()
                    except Exception:
                        pass
                elif name.endswith(".json"):
                    json_list.append(name)
            for jf in tqdm(json_list, desc=f"    {lz_path.name}", leave=False, mininterval=2):
                try:
                    data = json.loads(lz.read(jf))
                    for dlg in data["dataSet"]["dialogs"]:
                        ap = dlg["audioPath"].replace("KtelSpeech/", "", 1)
                        tp = dlg["textPath"].replace("KtelSpeech/", "", 1)
                        transcript = clean_main_transcript(txt_map.get(tp, ""))
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


# ── aux dataset (159) ─────────────────────────────────────────────────────────

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

    script_map: dict = {}
    with zipfile.ZipFile(txt_zip) as tz:
        for name in tqdm(tz.namelist(), desc="  Parsing text zip"):
            if not name.endswith(".txt"):
                continue
            content = tz.read(name).decode("utf-8", errors="ignore")
            sid, stn = None, None
            for line in content.splitlines():
                if line.startswith("scriptID"):
                    sid = line.split(":", 1)[1].strip()
                elif line.startswith("scriptITN"):
                    stn = line.split(":", 1)[1].strip()
            if sid and stn:
                script_map[sid] = clean_script_tn(stn)
    print(f"  Scripts: {len(script_map):,}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir",     default="/home/elicer/012.상담_음성_데이터")
    parser.add_argument("--aux_dir",      default="/home/elicer/159.숫자가_포함된_패턴_발화_데이터")
    parser.add_argument("--output_dir",   default="./data")
    parser.add_argument("--aux_ratio",    type=float, default=0.10)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--min_duration", type=float, default=0.3)
    parser.add_argument("--max_samples",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_train = process_main(args.main_dir, "train", args.min_duration, args.max_duration, args.max_samples)
    main_valid = process_main(args.main_dir, "valid", args.min_duration, args.max_duration, 0)
    aux_train  = process_aux(args.aux_dir, "train", args.min_duration, args.max_duration)
    aux_valid  = process_aux(args.aux_dir, "valid", args.min_duration, args.max_duration)

    print("\n--- Mixing train ---")
    train_records = mix(main_train, aux_train, args.aux_ratio, args.seed)
    print("--- Mixing valid ---")
    valid_records = mix(main_valid, aux_valid, args.aux_ratio, args.seed)

    write_jsonl(train_records, out / "train.jsonl")
    write_jsonl(valid_records, out / "valid.jsonl")
    print("\nDone.")


if __name__ == "__main__":
    main()
