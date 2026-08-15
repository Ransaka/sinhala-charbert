# Sinhala-CharBERT

**Dual-Channel Transformer for Sinhala Typo Detection, Correction, and Robust NLU**

---

## 1. Overview

Sinhala-CharBERT is a dual-channel Transformer architecture tailored for the Sinhala language. It processes input text simultaneously across two parallel streams:
1. **Token Channel (Subwords):** Captures semantic representation over BPE subword tokens.
2. **Character Channel (Phonological Units):** Captures phonetic and orthographic context over Akshara-level grapheme clusters extracted via `sinlib`.

The two channels interact layer-by-layer through a **Heterogeneous Interaction (HI) Module** utilizing multi-window 1D convolutions ($k \in [1, 3, 5]$) and GELU-activated divide projections with residual LayerNorm.

---

## 2. Architecture & Pipeline

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

---

## 3. SynTypo-SI Noise Synthesis Engine

The model is pre-trained using **SynTypo-SI**, a 4-stage probabilistic DAG modeling real-world typing and linguistic noise:
- **Stage 1 (Linguistic & Dialectal):** Regional morphology (Up-Country/Kandyan markers e.g., `යන්න` $\rightarrow$ `යන්ඩ`) and emphatic colloquial clitics.
- **Stage 2 (IME Simulation):** Wijesekara SLS 1134 physical keyboard 2D spatial drift $P_{\text{phys}}(k_j \mid k_i) \propto \exp(-d^2 / 2\sigma^2)$ and shift-inversion noise.
- **Stage 3 (Orthographic & Unicode):** Classical confusions (`න/ණ`, `ල/ළ`, `ච/ඡ`, `ක/ඛ`), composite vowel splitting (`ේ` $\rightarrow$ `ෙි`), and ZWJ stripping.
- **Stage 4 (Structural & Code-Switch):** Spacing anomalies, particle detachments, and conversational English insertion (`thanks`, `breaking news`).

---

## 4. Quick Start

### Installation

```bash
git clone <repo-url>
cd "CharBERT Sinhala"
uv sync
```

### Pre-Training CLI

```bash
uv run python scripts/train_charbert.py \
    --backbone_path "Ransaka/sinhala-bert-medium-v2" \
    --subword_tokenizer "Ransaka/sinhala-bert-medium-v2" \
    --dataset_name "Ransaka/sinhala-450M-sample" \
    --batch_size 32 \
    --max_steps 100000
```

### Running Tests

```bash
uv run pytest -v tests/
```
