# MIND-RX - Conversational Cognitive Enhancement System for Elderly with Dementia

## Overview
A conversational AI system designed for elderly patients with dementia.
Records daily speech, stores personalized memories, and induces natural recall conversations
to provide cognitive stimulation and emotional stability.
Responsible for Whisper fine-tuning for elderly speech recognition. (In progress)

## Tech Stack
Python, Whisper (OpenAI), HuggingFace Transformers, PyTorch, FastAPI, PostgreSQL, GPT-3.5-Turbo

## Key Features
- Voice recording and automatic summarization
- Personalized memory storage
- Cognitive stimulation conversation
- Sentiment analysis and responsive dialogue

## My Role (Fine-tuning)
- Fine-tuned Whisper model on elderly speech dataset (AI Hub)
- Resolved memory issues by converting data to Parquet format
- Applied gradient checkpointing for GPU memory optimization
- Ongoing: completing fine-tuning and API extraction

## Challenges & Solutions
- Memory shortage → Preprocessed data saved in .parquet format
- GPU memory issue → Increased evaluation interval in training code
- Storage shortage → Redirected Arrow cache to external HDD

## Status
Fine-tuning in progress — model accuracy still low, further training needed before API integration

## Period
2025.03.06~2025.07.22


---
---
---


# MIND-RX - 치매 노인을 위한 대화형 인지 향상 시스템

## 개요
치매 노인을 위한 대화형 AI 시스템.
일상 발화를 기반으로 기억을 기록하고, 자연스러운 회상 대화를 유도하여
인지 자극 및 정서적 안정을 제공.
Whisper 파인튜닝(노인 음성 인식) 파트 담당. (진행 중)

## 기술 스택
Python, Whisper (OpenAI), HuggingFace Transformers, PyTorch, FastAPI, PostgreSQL, GPT-3.5-Turbo

## 핵심 기능
- 음성 녹음 및 자동 요약
- 개인 맞춤 기억 저장
- 인지 자극 대화
- 감정 분석 및 반응형 대화

## 담당 역할 (파인튜닝)
- AI 허브 노인 음성 데이터를 활용한 Whisper 모델 파인튜닝
- 데이터 Parquet 형식 변환으로 메모리 문제 해결
- Gradient Checkpointing 적용으로 GPU 메모리 최적화
- 파인튜닝 완료 후 API 추출 진행 예정

## 문제 해결 과정
- 메모리 부족 → 데이터 전처리 파일 .parquet 형식으로 저장
- GPU 메모리 문제 → 평가 간격 늘리는 방식으로 코드 수정
- 저장 공간 부족 → 외장 HDD 연결해 arrow 캐시 저장

## 진행 현황
파인튜닝 진행 중 — 정확도 개선 후 API 연동 예정

## 기간
2025.03.06~2025.07.22
