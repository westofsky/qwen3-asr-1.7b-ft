# Qwen3-ASR-1.7B 한국어 Fine-tuning

## 환경 설정

```bash
cd /home/elicer/qwen3-asr-finetune
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 데이터 준비

```bash
# 빠른 테스트 (main train 5만 샘플만)
python prepare_dataset.py --max_samples 50000

# 전체 데이터 (수십 GB 디스크 필요)
python prepare_dataset.py
```

## 학습

```bash
python train.py --config config.yaml
```

## 모델 내보내기

```bash
python export_model.py --checkpoint ./checkpoints --output ./exported_model
```

## 다음 번 접속 시

```bash
cd /home/elicer/qwen3-asr-finetune
source venv/bin/activate
```

## 주의

- `scriptITN` 미사용. `scriptTN`의 `[...]` 브래킷 제거 후 사용.
- 보조 데이터(159)는 aux_ratio 기본 10%만 mixing.
