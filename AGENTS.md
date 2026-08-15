# Project Guidelines & Specification: Sinhala-CharBERT

This document outlines the rules, architectural concepts, noise synthesis pipelines, and development workflows that AI agents must strictly follow when contributing to the **Sinhala-CharBERT** typo detection and correction project.

---

## 1. Project Overview & Architecture

### 1.1 Goals
The objective is to implement and train **Sinhala-CharBERT**, a dual-channel Transformer designed for Sinhala typo correction and robust Natural Language Understanding (NLU). The framework used is **PyTorch** and the target language is Sinhala (`si`).

### 1.2 Dual-Channel Processing Concept
The architecture processes text through two parallel channels:
1. **Token Channel (Subwords):** Processes subword tokens generated via Byte-Pair Encoding (BPE) or WordPiece.
2. **Character Channel (Phonological Units):** Processes Akshara-level phonological units (grapheme clusters) extracted via the local `sinlib` library.

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
    N Layers    │      ┌────────────────────────────────────────────────────┐
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

## 2. Tokenization & Sequence Mapping Rules

### 2.1 Character & Phonological Unit Extraction
* **Engine:** Always use the `sinlib` package (`from sinlib import Tokenizer; from sinlib.utils.preprocessing import process_text, normalize_sinhala`) for phonological segmentations and string normalization.
* **Strategy:** Akshara-level clusters must be used (e.g., `'ආයුබෝවන්'` $\rightarrow$ `['ආ', 'යු', 'බෝ', 'ව', 'න්']`). Avoid raw Unicode code-point splits which isolate base letters from diacritics.
* **Special Tokens:** Handle special tokens (padding, unk, bos, eos) as parameterized configurations or using default `sinlib` parameters.

### 2.2 Sequence Alignment
* **Mapping Logic:** Map each subword token $w_i$ to its constituent sequence of phonological units $\{c_1^i, c_2^i, \dots, c_{n_i}^i\}$.
* **Boundary Tracking:** Track sequence indices $\text{start\_char\_idx}[i]$ and $\text{end\_char\_idx}[i]$ to map character channel embeddings back to token channel boundaries.

---

## 3. Modeling Components

### 3.1 Character Channel Bi-GRU Encoder
* Character embeddings are fed into a **Bidirectional GRU (Bi-GRU)** to capture local phonological context.
* **Boundary Concatenation:** For each subword token $w_i$, construct the token-aligned character embedding by concatenating the forward and backward hidden states of its boundary phonological units:
  $$h_i(x) = \left[ h_{\text{start\_char\_idx}[i]}(x) \;;\; h_{\text{end\_char\_idx}[i]}(x) \right]$$

### 3.2 Heterogeneous Interaction (HI) Module
A custom interaction module executed after every Transformer encoder block:
1. **Fusion Step:**
   * Linearly project the current token representation and token-aligned character representation.
   * Concatenate them along the hidden axis.
   * Apply a 1D Multi-Window Convolution across the token sequence (using parameterized multiple window sizes such as 1, 3, and 5) to integrate multi-scale context.
2. **Divide Step:**
   * Split the fused representation by projecting it back into separate token and character streams using GELU-activated feedforward projections.
   * Apply residual connections and Layer Normalization to obtain the updated token and character channel states.

---

## 4. Pre-Training Tasks & Advanced Noise Generation Pipeline (SynTypo-SI)

### 4.1 Dual-Objective Loss
Train the network using a combined objective function:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NLM}}$$

* **Masked Language Modeling (MLM):** Target Token Channel. Randomly mask tokens (mask token, random token, original token) and optimize with Cross-Entropy Loss.
* **Noisy Language Modeling (NLM):** Target Character Channel. Predict correct words from a dictionary of frequent Sinhala words using character representations subject to the SynTypo-SI noise pipeline.

### 4.2 SynTypo-SI Multi-Stage Noise Generation Pipeline
The noise synthesis pipeline operates as a multi-stage probabilistic DAG modeling physical keystroke faults, grapheme corruptions, encoding errors, dialectal shifts, and code-mixed noise:

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
|                                                                                    |
|           +---------------------------------+----------------------------------+   |
|           | Branch A (Wijesekara Physical)  | Branch B (Singlish Translit.)    |   |
|           | - QWERTY coordinate distance    | - Latin phoneme perturbations    |   |
|           | - Shift-state drop / mismatch   | - Ambiguous mapping expansion    |   |
|           | - Direct glyph substitution     | - Transliteration to Sinhala     |   |
|           +---------------------------------+----------------------------------+   |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Stage 3: Orthographic, Glyph & Unicode Corruption Layer                            |
| - Unicode decomposition & combining mark sequencing (e.g., ෙ + ි vs ේ)             |
| - ZWJ (Zero-Width Joiner) drop / misplacement (e.g., Rakaranshaya / Yanshaya)       |
| - Classical confusions (Mahaaprana, Sanyaka, Dental vs. Retroflex)                 |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Stage 4: Structural, Punctuation & Code-Switch Injection Layer                     |
| - Punctuation & space mutations (skipping space after full stop, full stop as space)|
| - Whitespace perturbation (particle detachment, token fusion)                      |
| - English code-switch injection & UI artifact insertion                            |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Output: Paired Training Tuple (Source_Noisy, Target_Clean, Edit_Metadata)          |
+------------------------------------------------------------------------------------+
```

#### 4.2.1 Stage 1: Linguistic & Dialectal Transformation Layer
* **Infinitive & Imperative Shifts:** `-න්න` $\rightarrow$ `-න්ඩ`, `-න්නට` $\rightarrow$ `-න්ට` (e.g., `යන්න` $\rightarrow$ `යන්ඩ`, `දෙන්න` $\rightarrow$ `දෙන්ඩ`).
* **Participle Truncation:** `කරලා` $\rightarrow$ `කරල`.
* **Emphatic Clitics:** `තමයි` $\rightarrow$ `තමා` / `තමෙයි`.

#### 4.2.2 Stage 2: IME & Keyboard Keystroke Noise
1. **Wijesekara Physical Layout Simulation (SLS 1134 Standard):**
   * **2D Euclidean Key Distance Kernel:** Spatial drift modeled across physical key coordinates $(x, y)$:
     $$d(k_i, k_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$
     Transition probability under typing inaccuracy variance $\sigma^2$:
     $$P_{\text{phys}}(k_j \mid k_i) = \frac{\exp\left(-\frac{d(k_i, k_j)^2}{2\sigma^2}\right)}{\sum_{k \in \mathcal{N}(k_i)} \exp\left(-\frac{d(k_i, k)^2}{2\sigma^2}\right)}$$
   * **Shift Modifier Drop / Inversion:** Simulates unshifted key hit when shifted was intended on the same key (e.g., Shift+`s` produces `ශ`, unshifted `s` produces `ි`; Shift+`f` produces `ේ`, unshifted `f` produces `ෙ`).
2. **Singlish Phonetic Transliteration Noise:**
   * **Ambiguity Noise:** Latin digraph variations (`t` $\rightarrow$ `th`, `d` $\rightarrow$ `dh`).
   * **Vowel Length Variations:** `aa` $\leftrightarrow$ `a`, `ee` $\leftrightarrow$ `i`, `oo` $\leftrightarrow$ `u`.
   * **Consonant Doubling Glitches:** `kanna` $\rightarrow$ `kana` (`කන්න` $\rightarrow$ `කන`).

#### 4.2.3 Stage 3: Orthographic, Glyph & Unicode Corruption Layer
1. **Classical & Orthographic Confusion:**
   * *Murdhaja / Dantaja Confusion:* (`'න'` $\leftrightarrow$ `'ණ'`), (`'ල'` $\leftrightarrow$ `'ළ'`).
   * *Mahaaprana / Alpaprana Confusion:* (`'ඛ'` $\leftrightarrow$ `'ක'`), (`'ඝ'` $\leftrightarrow$ `'ග'`), (`'ඡ'` $\leftrightarrow$ `'ච'`), (`'ඨ'` $\leftrightarrow$ `'ට'`), (`'ඪ'` $\leftrightarrow$ `'ඩ'`), (`'ථ'` $\leftrightarrow$ `'ත'`), (`'ධ'` $\leftrightarrow$ `'ද'`), (`'ඵ'` $\leftrightarrow$ `'ප'`), (`'භ'` $\leftrightarrow$ `'බ'`).
   * *Sanyaka (Prenasalized) Substitution:* (`'ඟ'` $\rightarrow$ `'ග'`), (`'ඳ'` $\rightarrow$ `'ද'`), (`'ඬ'` $\rightarrow$ `'ඩ'`), (`'ඹ'` $\rightarrow$ `'බ'`).
   * *Anusvara vs Nasal Confusion:* (`'ං'` $\leftrightarrow$ `'න්'`, `'ම්'`).
   * *Sibilants:* (`'ස'` $\leftrightarrow$ `'ශ'`, `'ෂ'`).
2. **Unicode Decomposition & Illegal Diacritic Sequencing:**
   * Composite vowel splits (e.g., `'ේ'` $\rightarrow$ `'ෙ'` + `'ි'`, `'ෝ'` $\rightarrow$ `'ෙ'` + `'ා'`).
   * Diacritic mutation (e.g., `'ුරු'` $\rightarrow$ `'ුරැ'`).
3. **ZWJ (Zero-Width Joiner) Sequence Stripping:**
   * Dropping `\u200D` in ligatures causing broken renderings for Rakaranshaya (`'ක්ර'` instead of `'ක්‍ර'`), Yanshaya (`'ක්ය'` instead of `'ක්‍ය'`), and Bandi Akuru (`'ක්ව'` instead of `'ක්‍ව'`).

#### 4.2.4 Stage 4: Structural, Punctuation & Code-Switch Injection Layer
1. **Punctuation & Space Mutations:**
   * **Skipping Space After Full Stop:** Removing space following punctuation (e.g., `". ගියා"` $\rightarrow$ `".ගියා"`).
   * **Full Stop As Space Delimiter:** Substituting space delimiters with full stops (e.g., `"මම ගෙදර ගියා"` $\rightarrow$ `"මම.ගෙදර.ගියා"`).
   * **Particle Detachment:** Accidental space insertion prior to case clitics/postpositions (`ට`, `ගේ`, `ගෙන්`, `ද`, `නම්`, `මයි`, `වත්`), e.g., `"තෙපිට"` $\rightarrow$ `"තෙපි ට"`.
   * **Token Fusion (Whitespace Drop):** Removing space between tokens/punctuation (e.g., `"comment: අප්පද"` $\rightarrow$ `"commentඅප්පද"`).
2. **Code-Switching Noise:**
   * Probabilistic insertion of English conversational markers, social-media prefixes/suffixes, and UI artifacts sampled from a configurable lexicon (e.g., `"thanks"`, `"siyatha news"`, `"Read more"`, `"Breaking news"`, `"Fake news"`, `"Good job"`, `"WTF"`).

### 4.3 Training Noise Curriculum
When training the model, use a phased noise schedule:
* **Phase 1 (Warmup):** Predominantly simple orthographic substitutions (`න/ණ`, `ල/ළ`, `ච/ඡ`).
* **Phase 2 (Complex Keystroke & Ligature Noise):** Introduce physical Wijesekara key-distance noise, shift drop/inversion, ZWJ deletions, and punctuation/spacing mutations.
* **Phase 3 (Robustness & Code-Switching):** Add code-switching markers, dialectal morphology, and particle splitting/fusions.

---

## 5. Backbone Initialization & Training Protocol

1. **Backbone Loading:** Load pre-trained weights from a partially trained or standard BERT checkpoint into the Token Embeddings, Position Embeddings, and Transformer Encoder layers.
2. **Random Initialization:** Initialize the character embedding, Bi-GRU, HI module, and NLM classification projection head randomly.
3. **Joint Training:** Resume pre-training under joint MLM + NLM loss using parameterized learning rate schedules with linear warmup and cosine decay.

---

## 6. Downstream Typo Correction Execution

* **Mode A: Bounded Word-Level Denoising:** Extract the character channel representation at corrupted token indices and decode using a classifier over the Sinhala word dictionary.
* **Mode B: Open-Vocabulary Seq2Seq Correction:** The CharBERT encoder outputs a fused sequence representation. An autoregressive Transformer decoder cross-attends to this state to generate corrected phonological units (`sinlib` tokens) sequentially.

---

## 7. Development Guidelines

* **Local Dependencies:** Always use the local `sinlib` codebase located at `/Users/ransaka/Study/sinlib`. Ensure that any virtual environment resolves to this directory or includes it in the path. Utilize `sinlib`'s `Tokenizer`, `process_text`, and `normalize_sinhala` for phonological segmentation and string normalization.
* **Code Standards:** Write modular, well-documented PyTorch code. All dimensions, loss weights, noise rates, and layer configurations must be parameterized in configuration objects rather than hardcoded.
* **No Emojis:** Do not use emojis in source files, commit messages, or comments.
* **Testing:** Prioritize testing the sequence alignment mappings, SynTypo-SI noise DAG stages (Wijesekara spatial kernel, punctuation anomalies, code-switch insertion), and HI Module tensor transformations.
