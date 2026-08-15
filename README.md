# Sinhala-CharBERT

**Dual-Channel Transformer for Sinhala Typo Detection, Correction, and Robust Natural Language Understanding**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dual-Channel Architecture](#2-dual-channel-architecture)
   - [Character Channel & Sequence Alignment](#21-character-channel--sequence-alignment)
   - [Boundary Pooling](#22-boundary-pooling)
   - [Heterogeneous Interaction (HI) Module](#23-heterogeneous-interaction-hi-module)
3. [SynTypo-SI Noise Synthesis Engine](#3-syntypo-si-noise-synthesis-engine)
   - [4-Stage Probabilistic DAG](#31-4-stage-probabilistic-dag)
   - [3-Phase Curriculum Training](#32-3-phase-curriculum-training)
4. [Installation & Setup](#4-installation--setup)
5. [Pre-Training Pipeline](#5-pre-training-pipeline)
   - [Quickstart Command](#51-quickstart-command)
   - [Hyperparameters & CLI Arguments](#52-hyperparameters--cli-arguments)
   - [Apple Silicon (MPS) & CUDA Optimization](#53-apple-silicon-mps--cuda-optimization)
6. [Downstream Typo Correction Execution](#6-downstream-typo-correction-execution)
   - [Mode A: Bounded Word-Level Denoising](#61-mode-a-bounded-word-level-denoising)
   - [Mode B: Open-Vocabulary Seq2Seq Decoder](#62-mode-b-open-vocabulary-seq2seq-decoder)
   - [Python API Usage](#63-python-api-usage)
   - [Interactive Terminal CLI](#64-interactive-terminal-cli)
   - [Fine-Tuning Mode B](#65-fine-tuning-mode-b)
7. [Repository Structure](#7-repository-structure)
8. [Running Test Suite](#8-running-test-suite)
9. [Citation & License](#9-citation--license)

---

## 1. Overview

Sinhala is a morphologically rich, Brahmic-family language characterized by complex orthography, nonlinear combining diacritics (*pili*), Zero-Width Joiner (ZWJ) ligatures (*bandi akuru*, *rakaranshaya*, *yanshaya*), and pronounced spoken-vs-written dialectal shifts. Standard subword tokenizers (Byte-Pair Encoding, WordPiece) break down under real-world Sinhala noisy text, leading to severe out-of-vocabulary (OOV) fragmentation.

**Sinhala-CharBERT** solves this by maintaining two synchronized representation streams:
1. **Token Channel (Subwords):** Preserves high-level contextual semantics over subword tokens.
2. **Character Channel (Phonological Units):** Models phonetic, orthographic, and keystroke representations over Akshara-level grapheme clusters extracted via `sinlib`.

---

## 2. Dual-Channel Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                    Input Sinhala Text                    │
                    └─────────────┬──────────────────────────────┬─────────────┘
                                  │                              │
                                  ▼                              ▼
                    ┌───────────────────────────┐  ┌───────────────────────────┐
                    │       BPE Subwords        │  │     sinlib Phono Units    │
                    └─────────────┬─────────────┘  └─────────────┬─────────────┘
                                  ▼                              ▼
                    ┌───────────────────────────┐  ┌───────────────────────────┐
                    │      Token Embedding      │  │      Char Embedding       │
                    └─────────────┬─────────────┘  └─────────────┬─────────────┘
                                  │                              ▼
                                  │                ┌───────────────────────────┐
                                  │                │     Sequence Bi-GRU       │
                                  │                │   (Boundary Concat State) │
                                  │                └─────────────┬─────────────┘
                                  │                              │
                                  ▼                              ▼
                       ┌────────────────────────────────────────────────────┐
                ┌─────►│          Transformer Encoder Layer (l)             │
                │      └──────────┬──────────────────────────────┬──────────┘
                │                 ▼                              ▼
    6 Layers    │      ┌────────────────────────────────────────────────────┐
                │      │      Heterogeneous Interaction Module (HI)         │
                │      │    - Step 1: Fusion (Linear Projections + CNN-1D)  │
                │      │    - Step 2: Divide (GELU Projection + Residual)   │
                └──────┴──────────┬──────────────────────────────┬──────────┘
                                  │                              │
                                  ▼                              ▼
                    ┌───────────────────────────┐  ┌───────────────────────────┐
                    │    Token Output Head      │  │     Char Output Head      │
                    │      (MLM Objective)      │  │      (NLM Denoising)      │
                    └───────────────────────────┘  └───────────────────────────┘
```

### 2.1 Character Channel & Sequence Alignment
* **Phonological Extraction:** Input strings are segmented into Akshara clusters using `sinlib` (e.g., `'ආයුබෝවන්'` $\rightarrow$ `['ආ', 'යු', 'බෝ', 'ව', 'න්']`).
* **Sequence Alignment Engine (`SequenceAlignmentEngine`):** Computes bidirectional coordinate mappings `[start_char_idx, end_char_idx]` matching each subword token $w_i$ to its corresponding phonological spans $\{c_{\text{start}}^i, \dots, c_{\text{end}}^i\}$.

### 2.2 Boundary Pooling
For each subword token $w_i$, the token-aligned character embedding $h_i$ is constructed by concatenating the forward hidden state of its first phonological unit and backward hidden state of its last phonological unit:
$$h_i = \left[ \vec{h}_{\text{start\_char\_idx}[i]} \;;\; \overleftarrow{h}_{\text{end\_char\_idx}[i]} \right]$$

### 2.3 Heterogeneous Interaction (HI) Module
Executed after every Transformer encoder block:
1. **Fusion Step:**
   * Linearly project and concatenate token states $T^{(l)}$ and character states $H^{(l)}$:
     $$M^{(l)} = [W_t T^{(l)} ; W_c H^{(l)}]$$
   * Apply 1D Multi-Window Convolutions ($k \in [1, 3, 5]$) across sequence length to capture multi-scale context.
2. **Divide Step:**
   * Project fused representation back into separate token and character streams:
     $$T^{(l+1)} = \text{LayerNorm}\left(T^{(l)} + \text{GELU}(W_{\text{div},t} M_{\text{conv}}^{(l)})\right)$$
     $$H^{(l+1)} = \text{LayerNorm}\left(H^{(l)} + \text{GELU}(W_{\text{div},c} M_{\text{conv}}^{(l)})\right)$$

---

## 3. SynTypo-SI Noise Synthesis Engine

### 3.1 4-Stage Probabilistic DAG

```
+------------------------------------------------------------------------------------+
|                               Input: Clean Sinhala Text                            |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Stage 1: Linguistic & Dialectal Transformation Layer                                |
| - Regional morphology / Up-Country (Kandyan) markers (e.g., "යන්න" -> "යන්ඩ")      |
| - Spoken-to-Written colloquialisms & emphatic clitics ("තමයි" -> "තමා")            |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Stage 2: Input Method Engine (IME) Simulation (Branching)                          |
| - Wijesekara SLS 1134 physical 2D Gaussian drift: P(k_j|k_i) ∝ exp(-d^2 / 2σ^2)    |
| - Shift-state drop / inversion (e.g., Shift+s 'ශ' -> unshifted 'ි')                |
| - Singlish Latin digraph variations (th/t, aa/a, ee/i)                             |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Stage 3: Orthographic, Glyph & Unicode Corruption Layer                            |
| - Classical confusions: Murdhaja/Dantaja (න/ණ, ල/ළ), Mahaaprana (ක/ඛ, ත/ථ)         |
| - Sanyaka prenasalized substitutions (ඟ->ග, ඳ->ද, ඬ->ඩ, ඹ->බ)                       |
| - Unicode decomposition & combining mark mutations (ේ -> ෙ + ි)                    |
| - Zero-Width Joiner (ZWJ) stripping in ligatures (ක්\u200Dර -> ක්ර)                |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Stage 4: Structural, Punctuation & Code-Switch Injection Layer                     |
| - Punctuation & spacing mutations (missing spaces, period as space delimiter)       |
| - Particle detachment (accidental space before 'ට', 'ගේ', 'ද', 'නම්')               |
| - English code-mixed conversational markers ("thanks", "breaking news")            |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Output: Paired Training Tuple (Source_Noisy, Target_Clean, Edit_Metadata)          |
+------------------------------------------------------------------------------------+
```

### 3.2 3-Phase Curriculum Training
Pre-training automatically shifts through three noise regimes:
* **Phase 1 (Warmup, 0% to 15%):** Simple orthographic swaps (`න/ණ`, `ල/ළ`, `ච/ඡ`).
* **Phase 2 (Complex Keystrokes & Ligatures, 15% to 50%):** Physical Wijesekara key distance drift, shift inversion, and ZWJ deletions.
* **Phase 3 (Full Robustness, 50% to 100%):** Adds spoken-to-written dialectal morphology, particle splits/fusions, and code-mixed noise.

---

## 4. Installation & Setup

### Prerequisites
* Python `>= 3.10`
* PyTorch `>= 2.0.0`
* `uv` (recommended) or `pip`

```bash
git clone https://github.com/Ransaka/sinhala-charbert.git
cd sinhala-charbert

# Using uv
uv sync

# Using standard pip
pip install -e .
```

---

## 5. Pre-Training Pipeline

Sinhala-CharBERT initializes from a pre-trained BERT backbone (such as `Ransaka/sinhala-bert-medium-v2`), loading subword embeddings and Transformer layers, while training the character Bi-GRU, HI modules, and Noisy Language Modeling (NLM) head from scratch under joint optimization:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NLM}}$$

### 5.1 Quickstart Command

```bash
uv run python scripts/train_charbert.py \
    --backbone_path "Ransaka/sinhala-bert-medium-v2" \
    --subword_tokenizer "Ransaka/sinhala-bert-medium-v2" \
    --dataset_name "Ransaka/sinhala-450M-sample" \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_steps 100000 \
    --output_dir "checkpoints/sinhala_charbert"
```

### 5.2 Hyperparameters & CLI Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--backbone_path` | `str` | `Ransaka/sinhala-bert-medium-v2` | Pre-trained BERT backbone weights. |
| `--dataset_name` | `str` | `Ransaka/sinhala-450M-sample` | HuggingFace dataset or local text corpus. |
| `--num_samples` | `int` | `None` | Max dataset rows to load (default: all). |
| `--batch_size` | `int` | `8` | Micro-batch size per device. |
| `--gradient_accumulation_steps` | `int` | `4` | Number of gradient accumulation steps (Effective batch size = $8 \times 4 = 32$). |
| `--max_subword_length` | `int` | `256` | Max sequence length for subword tokens. |
| `--max_char_length` | `int` | `512` | Max sequence length for phonological Aksharas. |
| `--learning_rate` | `float` | `5e-5` | Peak learning rate (with linear warmup and cosine decay). |
| `--max_steps` | `int` | `100000` | Total pre-training optimization steps. |
| `--save_steps` | `int` | `5000` | Frequency of checkpoint saves. |
| `--logging_steps` | `int` | `100` | Training metric logging interval. |

### 5.3 Apple Silicon (MPS) & CUDA Optimization
* **Mixed Precision (`fp16`):** Automatically enabled on CUDA devices via `torch.amp.GradScaler`.
* **Memory Management:** The trainer includes automatic periodic cache clearing (`torch.mps.empty_cache()`) and sequence padding caps to ensure stable training on Apple Silicon unified memory without OOM.

---

## 6. Downstream Typo Correction Execution

Sinhala-CharBERT provides two complementary operational modes:

### 6.1 Mode A: Bounded Word-Level Denoising
* **Mechanism:** Extracts character channel representations at corrupted token boundaries and scores candidates across the Sinhala frequent word lexicon (~20,000 words).
* **Strengths:** Ultra-fast inference; optimal for typos in common vocabulary.

### 6.2 Mode B: Open-Vocabulary Seq2Seq Decoder
* **Mechanism:** Autoregressive Transformer Decoder cross-attending to the CharBERT fused state $Z = \text{LayerNorm}(W_z [T ; H])$. Generates `sinlib` Akshara units sequentially.
* **Strengths:** Open vocabulary; repairs unseen words, names, slang, and intricate ligatures (`ක්‍ර`, `ශ්‍රී`, `ක්ෂ`).

### 6.3 Python API Usage

```python
from sinhala_charbert import SinhalaCharBERTCorrector

# 1. Load trained corrector
corrector = SinhalaCharBERTCorrector.from_pretrained(
    "checkpoints/sinhala_charbert/final_model",
    subword_tokenizer_name="Ransaka/sinhala-bert-medium-v2",
)

# 2. Correct a sentence
result = corrector.correct("මම ඉස්කොලෙ යන්ඩ ලැස්ති", mode="word_denoise")

print("Clean Text:", result.text)
# Output: "මම ඉස්කෝලේ යන්න ලෑස්ති"

# 3. View structured edit summary
print(result.summary())
# Output:
# Original : මම ඉස්කොලෙ යන්ඩ ලැස්ති
# Corrected: මම ඉස්කෝලේ යන්න ලෑස්ති
# Edits:
#   - [REPLACE] 'ඉස්කොලෙ' -> 'ඉස්කෝලේ' (orthographic)
#   - [REPLACE] 'යන්ඩ' -> 'යන්න' (dialectal_morphology)
#   - [REPLACE] 'ලැස්ති' -> 'ලෑස්ති' (orthographic)
```

### 6.4 Interactive Terminal CLI

Run the interactive CLI tool to test sentences directly:

```bash
# Interactive REPL mode
uv run python scripts/correct_text.py --checkpoint_path "checkpoints/sinhala_charbert/final_model" --mode "word_denoise"

# Single-sentence query
uv run python scripts/correct_text.py \
    --text "ශ්රී ලංකාවෙ මිනිස්සු කරුනාවන්තයි" \
    --checkpoint_path "checkpoints/sinhala_charbert/final_model" \
    --mode "word_denoise"
```

### 6.5 Fine-Tuning Mode B (Seq2Seq)

To fine-tune the open-vocabulary Transformer decoder on paired noisy/clean text:

```bash
uv run python scripts/train_seq2seq_corrector.py \
    --dataset_name "Ransaka/sinhala-450M-sample" \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_steps 10000 \
    --output_dir "checkpoints/sinhala_charbert_seq2seq"
```

---

## 7. Repository Structure

```
CharBERT Sinhala/
├── AGENTS.md                          # Project specification & guidelines
├── README.md                          # Repository documentation
├── pyproject.toml                     # Python dependencies & build config
├── data/
│   └── syntypo_sample_pairs.jsonl     # Sample synthetic noisy pairs
├── docs/
│   └── sequence_alignment.md          # Technical documentation on alignment
├── scripts/
│   ├── generate_synthetic_data.py     # SynTypo-SI synthetic data generation
│   ├── train_charbert.py              # Pre-training CLI (Joint MLM + NLM)
│   ├── train_seq2seq_corrector.py     # Mode B Seq2Seq fine-tuning CLI
│   └── correct_text.py                # Interactive terminal correction CLI
├── src/
│   └── sinhala_charbert/
│       ├── __init__.py                # Top-level API exports
│       ├── config/
│       │   ├── model_config.py        # SinhalaCharBERTConfig
│       │   ├── noise_config.py        # SynTypo-SI noise parameters & profiles
│       │   └── training_config.py     # TrainingConfig
│       ├── data/
│       │   ├── alignment.py           # SequenceAlignmentEngine
│       │   ├── char_tokenizer.py      # SinhalaCharTokenizer (sinlib wrapper)
│       │   ├── collator.py            # DualChannelDataCollator
│       │   ├── syntypo.py             # SinhalaTypoSynthesizer (4-stage DAG)
│       │   └── wijesekara.py          # SLS 1134 physical keyboard kernel
│       ├── models/
│       │   ├── char_encoder.py        # Sequence Bi-GRU with boundary pooling
│       │   ├── embeddings.py          # Token & Character Embedding modules
│       │   ├── encoder.py             # Interleaved Transformer + HI stack
│       │   ├── heads.py               # MLM and NLM prediction heads
│       │   ├── hi_module.py           # Heterogeneous Interaction (HI) module
│       │   ├── modeling_charbert.py   # Backbone & PreTraining models
│       │   ├── pipeline.py            # SinhalaCharBERTCorrector pipeline
│       │   ├── seq2seq_decoder.py     # Mode B Transformer Decoder
│       │   └── word_corrector.py      # Mode A BoundedWordCorrector
│       └── training/
│           ├── curriculum.py          # NoiseCurriculumScheduler
│           ├── dataset.py             # Dynamic dual-channel pre-train dataset
│           ├── dictionary.py          # SinhalaNLMDictionary (frequent words)
│           └── trainer.py             # SinhalaCharBERTTrainer
└── tests/
    ├── test_alignment.py              # Alignment & tokenization tests
    ├── test_corrector.py              # Mode A, Mode B & pipeline tests
    ├── test_imports.py                # Import & config sanity checks
    ├── test_models.py                 # Core model & backbone tests
    ├── test_syntypo.py                # SynTypo-SI DAG & keyboard tests
    └── test_training.py               # Pre-training loop & curriculum tests
```

---

## 8. Running Test Suite

Run the full automated unit test suite (24 tests across all modules):

```bash
uv run pytest -v tests/
```

To run a specific module test:

```bash
uv run pytest -v tests/test_corrector.py
uv run pytest -v tests/test_syntypo.py
uv run pytest -v tests/test_models.py
```

---

## 9. Citation & License

This project is licensed under the **MIT License**.

```bibtex
@misc{sinhala_charbert2026,
  author = {Ransaka Ravihara},
  title = {Sinhala-CharBERT: Dual-Channel Transformer for Sinhala Typo Detection and Robust NLU},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Ransaka/sinhala-charbert}}
}
```
