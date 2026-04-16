# Multilingual Toxicity Detection

Binary toxicity classification across English, German, and Finnish using fine-tuned transformer models and a TF-IDF baseline.

## Task

Given a comment, predict whether it is toxic (1) or non-toxic (0). The dataset contains text in three languages with the training split being predominantly English.

## Models

| Model | Training data |
|-------|--------------|
| TF-IDF + Logistic Regression | Original |
| TF-IDF + Logistic Regression | Augmented |
| XLM-RoBERTa (`xlm-roberta-base`) | Original |
| XLM-RoBERTa (`xlm-roberta-base`) | Augmented |
| mBERT (`bert-base-multilingual-cased`) | Original |

## Results (dev set)

### Overall

| Model | F1 | Accuracy | AUROC |
|-------|----|----------|-------|
| TF-IDF + LR | 0.875 | 0.878 | 0.919 |
| TF-IDF + LR (aug) | 0.875 | 0.878 | 0.937 |
| XLM-RoBERTa | 0.912 | 0.912 | 0.967 |
| XLM-RoBERTa (aug) | 0.913 | 0.913 | 0.969 |
| mBERT | 0.907 | 0.907 | 0.965 |

### Per language

| Model | EN F1 | DE F1 | FI F1 |
|-------|-------|-------|-------|
| TF-IDF + LR | 0.913 | 0.675 | 0.119 |
| TF-IDF + LR (aug) | 0.910 | 0.710 | 0.377 |
| XLM-RoBERTa | 0.940 | 0.767 | 0.784 |
| XLM-RoBERTa (aug) | 0.944 | 0.764 | 0.790 |
| mBERT | 0.941 | 0.747 | 0.448 |

XLM-RoBERTa trained on augmented data performs best overall and generalises most consistently across languages, particularly Finnish where the baseline almost completely fails.

## Notebooks

- `mvp_toxicity_detection_anon.ipynb` — data loading, preprocessing, baseline training, transformer fine-tuning
- `evaluation_anon.ipynb` — loads all saved models, runs inference, produces the comparison tables above

## Data

The dataset contains comments labelled for toxicity. Dev and test splits are balanced across English (`en`), German (`de`), and Finnish (`fi`).

Text preprocessing for the baseline uses language-aware lemmatisation and stopword removal via spaCy (`en_core_web_sm`, `de_core_news_sm`, `fi_core_news_sm`). Transformer models use raw text with the model's own tokenizer.
