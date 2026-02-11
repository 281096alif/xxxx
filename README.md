# Medical SOAP Notes Generation using BART

This project fine-tunes `facebook/bart-base` on a medical dialogue dataset to generate SOAP (Subjective, Objective, Assessment, Plan) notes. SOAP notes are the standard method of documenting patient encounters in clinical settings, and automating their generation from doctor-patient dialogues can significantly reduce administrative burden and improve clinical workflow efficiency.

## Model Performance

The fine-tuned model is trained on a local NVIDIA RTX 4060 Ti (8GB VRAM) and achieves the following metrics on the test set:

- **ROUGE-1**: 67.33
- **ROUGE-2**: 40.16
- **ROUGE-L**: 48.76

These scores represent a substantial improvement over the untrained baseline model.

## What are SOAP Notes?

SOAP notes are structured clinical documentation consisting of four components:
- **Subjective**: Patient's symptoms and complaints
- **Objective**: Clinical findings and observations
- **Assessment**: Diagnosis or medical evaluation
- **Plan**: Treatment plan and follow-up actions

## Dataset

The dataset consists of doctor-patient dialogues paired with corresponding SOAP notes. It is split into:

* **Train**: 9,250 examples
* **Validation**: 500 examples
* **Test**: 250 examples

### Data Files

Files are expected in the root directory:

* `medical_dialogue_train.csv`
* `medical_dialogue_validation.xlsx`
* `medical_dialogue_test.xlsx`

### Dataset Structure

Each example contains:
- Doctor-patient dialogue (input)
- Corresponding SOAP note (target output)


## Model Information

| Property | Value |
|----------|-------|
| **Base Model** | `facebook/bart-base` |
| **Architecture** | 6 encoder + 6 decoder layers, 768 hidden |
| **Parameters** | ~139M |
| **Tokenizer** | BART tokenizer (vocab size 50,265) |
| **Fine-tuned for** | Conditional generation of SOAP notes |
| **Hardware** | NVIDIA RTX 4060 Ti (8GB) |
| **Mixed Precision** | FP16 enabled |
| **Gradient Checkpointing** | Enabled (VRAM saving) |

### Why BART-base?

BART (Bidirectional and Auto-Regressive Transformer) is a denoising autoencoder pre-trained by corrupting text and learning to reconstruct it. Its encoder-decoder structure is ideal for abstractive summarization tasks. BART-base offers a strong balance between performance and computational cost – it fits comfortably on a consumer-grade GPU after optimizations.


## Fine-Tuning Process

### 1. Thought Process & Approach

The primary challenge was to adapt a general-purpose summarization model to the highly structured, domain-specific SOAP format while respecting hardware constraints. The following steps were carefully designed:

#### 📊 Data Analysis

Initial exploratory analysis of the dataset revealed:
* **Dialogue length** – 99th percentile ~752 tokens
* **SOAP note length** – 99th percentile ~486 tokens

These values were used as `max_input_length` and `max_target_length` to avoid unnecessary truncation or wasteful padding.

#### ⚠️ Complexity 1: Data Collator Errors

Early attempts used dynamic padding with Hugging Face's `DataCollatorForSeq2Seq`. However, the notebook environment exhibited persistent `KeyError` and type mismatches because the raw text columns (`Dialogue`, `SOAP`) remained in the dataset after tokenization.

**Solution:** Explicitly tokenize with static padding (`padding='max_length'`) and remove all columns except `input_ids`, `attention_mask`, `labels`. This allowed the simple `default_data_collator` to work flawlessly.

#### 🧠 Complexity 2: ROUGE Computation Robustness

The `evaluate.load('rouge')` library returns scores in different formats depending on the version (object with `.mid`, direct float, or dict). Additionally, `-100` labels must be replaced before decoding, and predictions from `model.generate` must be carefully cleaned.

**Solution:** A custom `compute_metrics` function that:
* Converts tensors to numpy
* Replaces `-100` with `pad_token_id`
* Implements a `safe_decode` that filters out-of-vocabulary and `None` values
* Handles all ROUGE return formats gracefully

#### 💾 Complexity 3: VRAM Constraints on RTX 4060 Ti (8GB)

BART-base usually requires ~6GB for a batch size of 8. With long sequences (752/486 tokens), even batch size 4 caused OOM.

**Solution:**
* **Gradient checkpointing** – trades compute for memory, reduces activation memory by ~50%
* **Gradient accumulation** – set to 2, effective batch size = 8 while using only 4 per device
* **FP16 mixed precision** – halves memory footprint
* **Static padding** – avoids on-the-fly padding overhead

With these settings, training runs comfortably at 4 samples/GPU.

#### 🚀 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Learning rate** | 3e-5 |
| **Batch size (per device)** | 4 |
| **Gradient accumulation** | 2 |
| **Effective batch size** | 8 |
| **Epochs** | 5 |
| **Optimizer** | AdamW |
| **Weight decay** | 0.01 |
| **Warmup steps** | 0 |
| **Evaluation strategy** | every 2000 steps |
| **Save strategy** | every 2000 steps |
| **Generation beams** | 4 |
| **FP16** | Yes |

Training took **~1h 23min** for 5 epochs (5780 steps).

### 2. Libraries Used

* `transformers` – BART model, tokenizer, Seq2SeqTrainer
* `datasets` – Hugging Face Dataset API for efficient mapping
* `evaluate` – ROUGE metric with version-agnostic parsing
* `pandas` / `openpyxl` – data loading and cleaning
* `torch` – CUDA management, gradient checkpointing
* `tqdm` – progress bars for inference
* `accelerate` – (implicitly) used by trainer for mixed precision

## 📈 Evaluation Results

### Test Set ROUGE Scores

| Metric | Score (F1, %) |
|--------|---------------|
| **ROUGE-1** | 67.33 |
| **ROUGE-2** | 40.16 |
| **ROUGE-L** | 48.76 |
| **ROUGE-Lsum** | 59.18 |

These scores represent the fine-tuned model evaluated on the 250 unseen test dialogues.

### Comparison with Pre-trained Baseline

The untrained `facebook/bart-base` produces dialogue repetition, irrelevant conversation, and no SOAP structure (see notebook output). In contrast, the fine-tuned model:

* Follows the SOAP format (S / O / A / P sections) consistently
* Extracts key clinical facts accurately
* Generates coherent treatment plans
* Handles rare diseases (e.g., mycobacterial peritonitis) by correctly synthesizing information from the dialogue

**Qualitative Example:** For a dialogue about tuberculosis, the fine-tuned model correctly lists the four-drug regimen, while the baseline repeats the conversation verbatim.


