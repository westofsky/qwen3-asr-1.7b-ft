# Qwen3-ASR-1.7B 한국어 Fine-tuning (Asterisk 16k 전화망 타깃)

## 데이터셋

| 역할 | 디렉터리 | 오디오 | 라벨 전처리 |
|------|---------|--------|-------------|
| main | `007.저음질_전화망_음성인식_데이터` | WAV 8kHz/16bit/mono | `o/ n/ b/ l/` 제거, `(A)/(B) → A`, `(())` 제거 |
| aux  | `159.숫자가_포함된_패턴_발화_데이터` | PCM 16kHz/16bit/mono | `scriptITN` 사용 (아라비아 숫자 표기) |

## Asterisk 16k 타깃 전략

- **007 (main)**: 이미 8kHz 전화망 원천. 로더에서 16k로 업샘플만 하면 Asterisk 포맷과 자연히 매칭.
- **159 (aux)**: 16kHz clean. 학습 시 `phone_band_filter` (16k→8k→16k 라운드트립)로 4kHz 대역제한을 걸어 Asterisk narrowband 신호에 맞춤. `aux_phone_prob=1.0` 기본.

## 환경 설정

```bash
cd /home/tantancore/qwen3-asr-1.7b-ft
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 데이터 준비

```bash
# 빠른 테스트 (main train 5만 샘플만)
python prepare_dataset.py --max_samples 50000

# 전체 데이터
python prepare_dataset.py
```

생성 결과: `./data/train.jsonl`, `./data/valid.jsonl`

## 학습

```bash
# 기본값 (aux 100% phone-band filter)
python train.py --max_steps 5000

# aux 50%만 전화망 시뮬, 나머지는 wideband로 (일반 16k 마이크 소스도 섞어 학습)
python train.py --aux_phone_prob 0.5

# main에도 추가 증강 (매우 노이지한 환경 대응)
python train.py --main_phone_prob 0.3
```

## 모델 내보내기

```bash
python export_model.py --checkpoint ./checkpoints --output ./exported_model
```

## 추론 (Asterisk에서 받은 16k WAV/PCM)

```python
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained("./exported_model")
result = model.transcribe("asterisk_recording.wav")
print(result)
```

## 주의

- 007의 JSON `dataSet.dialogs[].text`를 바로 사용 (별도 `textPath` 없음).
- 159의 `scriptTN`(대괄호 포함 한글 읽기) 대신 `scriptITN`(207동 / 3시 등 숫자 표기)을 학습.
- `phone_band_filter`는 librosa resample 라운드트립. μ-law 코덱 시뮬까지 필요한 경우 별도 확장.
