# Transactional Voice AI

This repository contains the code for Transactional Voice AI - a system for building and deploying voice based conversational AI systems, powered by models from AI4Bharat.

For enabling voice based conversations, Transactional Voice AI is build up from following three modules:
- ASR- Use AI4Bharat's ASR models to transcribe audio input to text
- Intent Recognition- Transcribed text is used to detect intent by using an intent classifier built on top of IndicBERT
- Entity Recognition- Extraction of various entity type is supported by using regex based matching algorithms