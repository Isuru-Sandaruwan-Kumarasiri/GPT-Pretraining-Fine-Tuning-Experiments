
# GPT Pretraining & Fine-Tuning Experiments

A collection of hands-on experiments covering the full lifecycle of language models — from **pretraining a GPT from scratch**, to **parameter-efficient fine-tuning (LoRA/QLoRA)**, **preference alignment (DPO)**, and a set of **tokenization / word-embedding pipelines** applied to a Named Entity Recognition (NER) task. Everything is implemented in PyTorch/HuggingFace and designed to run on Google Colab.

## Repository Structure

```
GPT-Pretraining-Fine-Tuning-Experiments/
├── Pretraining_nanoGPT/
│   ├── nanoGPT_pretraining.ipynb
│   └── Pre-training NanoGPT.txt
├── Lora_fine_tuning/
│   └── LLM(Qwen)_fine_tuning.ipynb
├── DPO/
│   └── DPO.ipynb
└── experiments_tockenizations_embedding/
    ├── tokenizers.ipynb
    ├── Word2Vec_Complete_Pipeline.ipynb
    ├── GloVe_Pipeline.ipynb
    ├── FastText_Pipeline.ipynb
    ├── BERT_Pipeline.ipynb
    ├── ELMo_Pipeline.ipynb
    └── NER_Report_S20426.pdf
```

---

## 1. `Pretraining_nanoGPT/` — GPT Pretraining From Scratch

A minimal, [nanoGPT](https://github.com/karpathy/nanoGPT)-style, decoder-only Transformer implemented and trained **from scratch** in pure PyTorch (no pretrained weights).

**What it does:**
- Downloads the **Tiny Shakespeare** dataset and builds a character-level tokenizer (`stoi`/`itos` encode-decode).
- Implements a GPT model from first principles:
  - `CausalSelfAttention` (with PyTorch 2.0 Flash Attention when available, manual masked attention as fallback)
  - `MLP` (GELU feed-forward block)
  - `Block` (pre-norm transformer block: attention + MLP with residual connections)
  - `GPT` (token + positional embeddings, stack of blocks, weight-tied LM head)
- Configurable via a `GPTConfig` dataclass (`block_size`, `vocab_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`) — default config is 6 layers, 6 heads, 384 embedding dim.
- Trains with `AdamW` for a fixed number of iterations, periodically evaluating train/val loss.
- Implements autoregressive `generate()` with temperature and top-k sampling.
- Saves the trained model checkpoint and generates sample Shakespeare-style text from the trained model.

**Files:**
- `nanoGPT_pretraining.ipynb` — the main runnable notebook.
- `Pre-training NanoGPT.txt` — a plain-text export of the same pretraining code.

---

## 2. `Lora_fine_tuning/` — Parameter-Efficient Fine-Tuning with LoRA/QLoRA

Fine-tunes **Qwen2.5-0.5B-Instruct** on a custom Question-Answering dataset built from FIFA World Cup statistics, using **LoRA** (Low-Rank Adaptation) with 4-bit quantization (QLoRA).

**What it does:**
- Loads `WorldCups.csv`, `WorldCupMatches.csv`, and `WorldCupPlayers.csv` and programmatically generates Q&A pairs (tournament winners/hosts, match results, player appearances, etc.).
- Formats each example in a chat-instruction template:
  `<|system|>...</s><|user|>...</s><|assistant|>...</s>`
- Splits the generated dataset into train (80%) / validation (10%) / test (10%) sets.
- Loads the base model in 4-bit (`BitsAndBytesConfig`, NF4 quantization) and applies a `LoraConfig` on top via 🤗 PEFT.
- Fine-tunes with `SFTTrainer` / `SFTConfig` from the `trl` library (gradient accumulation, `paged_adamw_32bit` optimizer).
- Saves the resulting LoRA adapter for downstream use (this adapter is later consumed by the DPO notebook).

**Key libraries:** `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`.

---

## 3. `DPO/` — Preference Alignment with Direct Preference Optimization

Takes the **LoRA/SFT-fine-tuned Qwen2.5-1.5B-Instruct** model from the previous stage and further aligns it using **DPO (Direct Preference Optimization)** on the same FIFA World Cup domain.

**What it does:**
- Builds a **preference dataset**: for each question, generates a `chosen` (correct) and `rejected` (incorrect/verbose) response pair, covering winner/host questions, match results, and style preferences.
- Splits preference records into train/val/test (80/10/10).
- Loads the base model in 4-bit and attaches the previously trained SFT LoRA adapter (kept trainable) via `PeftModel`.
- Configures and runs `DPOTrainer`/`DPOConfig` from `trl` (β = 0.1, 2 epochs, `paged_adamw_32bit`, learning rate 5e-6).
- Plots training/validation loss and the reward margin (how well the model separates chosen vs. rejected responses) over training steps.
- Runs qualitative evaluation, comparing the **SFT-only** model against the **DPO-aligned** model on a battery of FIFA World Cup questions.

**Key libraries:** `transformers`, `peft`, `trl` (`DPOTrainer`), `bitsandbytes`, `evaluate`, `rouge_score`, `bert_score`, `matplotlib`.

---

## 4. `experiments_tockenizations_embedding/` — Tokenization & Embedding Pipelines for NER

A comparative study of different tokenization strategies and word-representation methods, all applied to the same downstream task: **Named Entity Recognition** (disease/chemical entity tagging on a BIO-tagged corpus).

| Notebook | Representation | Notes |
|---|---|---|
| `tokenizers.ipynb` | Whitespace / NLTK / WordPiece (BERT) | Compares tokenization strategies and aligns BIO tags to sub-word tokens; preprocesses raw CoNLL-format data into train/val/test splits. |
| `Word2Vec_Complete_Pipeline.ipynb` | Word2Vec | Custom PyTorch `Dataset`, training loop, and evaluation routines for a Word2Vec-based NER tagger. |
| `GloVe_Pipeline.ipynb` | GloVe | Loads pretrained GloVe vectors and trains a downstream sequence tagger (BiLSTM-based). |
| `FastText_Pipeline.ipynb` | FastText | Same pipeline structure, using FastText embeddings (handles OOV via subword n-grams). |
| `BERT_Pipeline.ipynb` | BERT (contextual) | Uses `BertModel`/`BertTokenizer` to extract contextual embeddings, then trains a downstream classifier/tagger. |
| `ELMo_Pipeline.ipynb` | ELMo (contextual) | BiLSTM tagger on top of (mock/precomputed) ELMo vectors. |
| `NER_Report_S20426.pdf` | — | Write-up summarizing and comparing results across all the embedding methods above. |

**Common pattern across pipelines:**
- Custom `Dataset` class (e.g. `NERDataset`, `ELMoDataset`) that loads pre-tokenized/pre-tagged data.
- `collate_fn` for padding variable-length sequences (`pad_sequence`) with `-100` used as the ignore-index for tag padding.
- A BiLSTM (or transformer-based) tagger head on top of the chosen embeddings.
- Evaluation via `sklearn` (`confusion_matrix`, `classification_report`) and visualization with `matplotlib`/`seaborn`.

---

## Setup & Requirements

All notebooks were developed for **Google Colab** (they mount Google Drive for data/checkpoint storage) with GPU acceleration. To run locally, adjust the `drive.mount(...)` and hard-coded `/content/drive/...` paths to your own data locations.

Core dependencies (install as needed per notebook):

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes evaluate
pip install rouge_score bert_score
pip install scikit-learn matplotlib seaborn nltk tiktoken wandb tqdm
```

- **GPU recommended** for `Lora_fine_tuning/`, `DPO/`, and the embedding pipelines (BERT/ELMo in particular).
- `Pretraining_nanoGPT/` is small enough to run on CPU (slowly) or a single Colab GPU.

## Suggested Order

If you're reading through the experiments end-to-end, this mirrors a typical LLM training pipeline:

1. **`Pretraining_nanoGPT/`** — understand the base Transformer architecture and pretraining from scratch.
2. **`experiments_tockenizations_embedding/`** — explore how different tokenization/embedding choices affect a downstream NLP task.
3. **`Lora_fine_tuning/`** — take a pretrained model and adapt it efficiently to a custom domain (FIFA World Cup Q&A) via LoRA/QLoRA.
4. **`DPO/`** — further align the fine-tuned model's outputs to human/preference-style judgments via DPO.

## Notes

- Datasets referenced (`WorldCups.csv`, `WorldCupMatches.csv`, `WorldCupPlayers.csv`, CoNLL-format NER files) are **not included** in this repository — they are expected to be present in Google Drive at the paths configured in each notebook.
- Model checkpoints and LoRA adapters are saved to Google Drive paths and are also not included in the repo.
