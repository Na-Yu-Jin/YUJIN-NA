import os
import shutil
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from jiwer import wer

# 임시 디렉토리 설정을 가장 먼저 처리
TMP_PATH = "tmp"
tempfile.tempdir = TMP_PATH
os.environ["TEMP"] = TMP_PATH
os.environ["TMP"] = TMP_PATH
os.environ["PYARROW_TEMP_DIR"] = TMP_PATH
pa.set_cpu_count(1)

print("🤪 현재 임시파일 경로:", tempfile.gettempdir())

start_time = time.time()

# 설정 경로
BASE_PATH = "hf_cache"
PARQUET_PATH = os.path.join(BASE_PATH, "parquet_dataset")
CHUNK_PREFIX = os.path.join(BASE_PATH, "parquet_chunk_")
CHUNK_SIZE = 100_000
ARROW_CACHE_PATH = "hf_arrow_cache"

# 디렉토리 생성
os.makedirs(TMP_PATH, exist_ok=True)
os.makedirs(PARQUET_PATH, exist_ok=True)
os.makedirs(ARROW_CACHE_PATH, exist_ok=True)

# 환경변수 설정
os.environ["HF_HOME"] = BASE_PATH
os.environ["HF_DATASETS_CACHE"] = ARROW_CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = BASE_PATH
os.environ["HF_HUB_CACHE"] = BASE_PATH

# 데이터 불러오기 및 오디오 캐스팅
data_files = {
    "train": "data/train.csv",
    "validation": "data/dev.csv",
    "test": "data/test.csv",
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",", cache_dir=ARROW_CACHE_PATH)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

MODEL_NAME = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir=BASE_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=BASE_PATH).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

bad_paths = []
MAX_LABEL_LENGTH = 448

def prepare_dataset(batch):
    input_features, labels, valid_flags = [], [], []
    for audio, transcription in zip(batch["audio"], batch["transcription"]):
        try:
            features = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            label_ids = processor.tokenizer(transcription).input_ids[:MAX_LABEL_LENGTH]
            input_features.append(features)
            labels.append(label_ids)
            valid_flags.append(True)
        except Exception as e:
            path_str = audio.get("path", "<unknown>")
            print(f"⚠️ 건너뛰: {path_str} / 예외: {e}")
            bad_paths.append(path_str)
            input_features.append(None)
            labels.append(None)
            valid_flags.append(False)
    return {"input_features": input_features, "labels": labels, "valid": valid_flags}

# parquet 저장 경로로 전처리 및 저장
train = dataset["train"]
total = len(train)
chunks = [train.select(range(i, min(i + CHUNK_SIZE, total))) for i in range(0, total, CHUNK_SIZE)]
processed_chunks = []

for i, chunk in enumerate(chunks):
    path = f"{CHUNK_PREFIX}{i}.parquet"
    if os.path.exists(path):
        print(f"✅ parquet chunk {i} 존재함 → 로드")
        processed = load_dataset("parquet", data_files=path, cache_dir=ARROW_CACHE_PATH, keep_in_memory=False)["train"]
    else:
        print(f"🧹 chunk {i} 처리 중...")
        processed = chunk.map(prepare_dataset, batched=True, batch_size=8, remove_columns=chunk.column_names)
        processed = processed.filter(lambda x: x["valid"])
        processed.to_parquet(path)
    processed_chunks.append(processed)

merged_train = concatenate_datasets(processed_chunks)
eval_dataset = dataset["validation"].map(prepare_dataset, batched=True, batch_size=8, remove_columns=dataset["validation"].column_names)
eval_dataset = eval_dataset.filter(lambda x: x["valid"])

train_dataset = merged_train

with open("bad_files.txt", "w", encoding="utf-8") as f:
    for p in bad_paths:
        f.write(p + "\n")

class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [f["input_features"] for f in features]
        label_features = [f["labels"] for f in features]
        return {
            "input_features": torch.tensor(input_features),
            "labels": self.processor.tokenizer.pad({"input_ids": label_features}, padding=True, return_tensors="pt").input_ids,
        }

def compute_metrics(pred):
    pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score, "accuracy_percent": (1 - wer_score) * 100}

training_args = Seq2SeqTrainingArguments(
    output_dir="output/finetune",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=300,
    evaluation_strategy="steps",
    eval_steps=10000,
    save_steps=2000,
    logging_dir="output/finetune/logs",
    logging_steps=200,
    num_train_epochs=1,
    fp16=True,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    report_to="none",
    predict_with_generate=False,
    generation_max_length=128
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=WhisperDataCollator(processor),
    compute_metrics=compute_metrics
)

checkpoints = sorted(Path(training_args.output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
resume_checkpoint = str(checkpoints[-1]) if checkpoints else None

if resume_checkpoint:
    print(f"🔁 체크포인트에서 재시작: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    print("🚀 새 학습 시작")
    trainer.train()

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

print(f"✅ 전체 학습 완료! 총 소요 시간: {(time.time() - start_time) / 60:.2f}분")
