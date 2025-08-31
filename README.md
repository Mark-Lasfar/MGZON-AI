---
title: MGZon Chatbot
emoji: "🤖"
colorFrom: "blue"
colorTo: "green"
sdk: docker
app_file: main.py
pinned: false
---

# MGZON-AI
  A versatile chatbot powered by MGZON/Veltrix for MGZon queries. Supports code generation, analysis, review, web search, and MGZon-specific queries. Licensed under Apache 2.0.




---
library_name: transformers
license: apache-2.0
🌐 **Live Demo**
 [Live Demo](https://huggingface.co/spaces/MGZON/mgzon-app)
base_model: MGZON/Veltrix
tags:
- generated_from_trainer
model-index:
- name: mgzon-flan-t5-base
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->


# MGZON/Veltrix

This model is a fine-tuned version of [MGZON/Veltrix](https://huggingface.co/MGZON/Veltrix) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: nan

## Features
- **Text Queries**: Ask anything and get detailed responses.
- **Audio Input/Output**: Record audio directly or convert text to speech.
- **Image Analysis**: Capture images from webcam or upload for analysis.
- **Web Search**: Enable DeepSearch for real-time web context.
- **API Support**: Use endpoints like `/api/chat`, `/api/audio-transcription`, `/api/text-to-speech`, `/api/image-analysis`.

## Setup
1. Add `HF_TOKEN` and `BACKUP_HF_TOKEN` as Secrets in Space settings.
2. Add `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` for web search (optional).
3. Set `PORT=7860`, `QUEUE_SIZE=80`, `CONCURRENCY_LIMIT=20` as Variables.
4. Ensure `requirements.txt` and `Dockerfile` are configured correctly.

## Usage
Access the app at `/gradio` or use API endpoints. Examples:
- **Text**: "Explain AI history."
- **Audio**: Record audio for transcription.
- **Image**: Capture or upload an image for analysis.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 2
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.2456        | 1.0   | 1488 | nan             |
| 0.0888        | 2.0   | 2976 | nan             |
| 15.9533       | 3.0   | 4464 | nan             |
| 0.1136        | 4.0   | 5952 | nan             |
| 0.0626        | 5.0   | 7440 | nan             |


### Framework versions

- Transformers 4.55.2
- Pytorch 2.8.0+cu126
- Datasets 4.0.0
- Tokenizers 0.21.4

