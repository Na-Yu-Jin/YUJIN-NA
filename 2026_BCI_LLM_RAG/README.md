# A Dual-Gate-Based Safe Medical Triage Pipeline for Mitigating BCI (EEG) Input Uncertainty and LLM Hallucination Risks

## Overview
A personal research project proposing a dual-gate safety pipeline that combines EEG-based BCI and medical LLM (Meditron-7B) to eliminate dangerous advice in high-risk medical triage environments. Conducted collaboratively with iterative peer review.

## Tech Stack
Python, PyTorch, HuggingFace, Meditron-7B, EEG Signal Processing, RAG, BitsAndBytes, MNE, MNE-BIDS, Scikit-learn, Sentence-Transformers

## Key Contributions
- Designed GATE1: P300-based BCI input reliability scoring (ACCEPT/UNSURE/REJECT) using block-level confidence
- Designed GATE2: MedQuAD-based RAG + Red-flag rules to prevent LLM hallucination
- Integrated Meditron-7B (4-bit quantized) as medical backbone LLM
- Achieved dangerous advice rate of 0 across both 10 and 100 test cases
- Conducted subject-specific threshold tuning via coverage-risk tradeoff

## Pipeline Structure
EEG Signal → GATE1 (BCI Input Reliability) → GATE2 (RAG + Red-flag) → Meditron-7B → Triage Output (EMERGENCY / NOT EMERGENCY / UNSURE)

## Dataset
- EEG: RSVP-based P300 Speller BCI dataset (Won et al., 2022) — 55 subjects, 32ch, 512Hz
- Medical QA: MedQuAD (NIH-based medical question answering dataset)

## Period
2025.12 ~ 2026.01

## Note
Google Drive paths in the code should be updated to match your own directory structure before running.


---
---
---


# BCI(EEG) 입력 불확실성과 LLM 환각 위험을 완화하는 이중게이트 기반 안전 의료 문진 파이프라인

## 개요
EEG 기반 BCI와 의료 LLM(Meditron-7B)을 결합하여 고위험 의료 환경에서 위험한 조언 발생률 0건을 달성한 이중 게이트 안전 파이프라인 개인 연구 프로젝트. 친구와 공동 연구로 상호 코드 리뷰하며 진행.

## 기술 스택
Python, PyTorch, HuggingFace, Meditron-7B, EEG Signal Processing, RAG, BitsAndBytes, MNE, MNE-BIDS, Scikit-learn, Sentence-Transformers

## 주요 기여
- GATE1 설계: P300 분류기 블록 단위 신뢰도 평가(ACCEPT/UNSURE/REJECT)로 불확실한 BCI 입력 차단
- GATE2 설계: MedQuAD 기반 RAG + Red-flag 규칙으로 LLM 환각 위험 방지
- Meditron-7B(4-bit 양자화) 의료 특화 백본 LLM 연동
- 10건 및 100건 테스트 모두 위험한 조언 발생률 0건 달성
- Coverage-risk tradeoff 기반 피험자별 임계값 튜닝

## 파이프라인 구조
EEG 신호 → GATE1 (BCI 입력 신뢰도) → GATE2 (RAG + Red-flag) → Meditron-7B → Triage 출력 (EMERGENCY / NOT EMERGENCY / UNSURE)

## 데이터셋
- EEG: RSVP 기반 P300 Speller BCI 데이터셋 (Won et al., 2022) — 55명, 32채널, 512Hz
- 의료 QA: MedQuAD (NIH 기반 의료 질의응답 데이터셋)

## 기간
2025.12 ~ 2026.01

## 주의사항
코드 내 구글 드라이브 경로는 본인 환경에 맞게 수정 후 실행해주세요.
