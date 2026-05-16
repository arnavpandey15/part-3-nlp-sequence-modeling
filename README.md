# Part 3 – NLP and Sequence Modeling Mini Project

> **Course**: Applied Neural Networks, Computer Vision, NLP, and AI Solution Design  
> **Dataset**: Customer Support Text Classification (1,500 records · 3-class sentiment)  
> **Framework**: TensorFlow 2 / Keras · scikit-learn · Pandas · NLTK-free preprocessing

---

## Repository Structure

```
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb                         ← All 6 tasks (runnable end-to-end)
├── requirements.txt
├── customer_support_text_classification.csv
└── results/
    ├── evaluation_outputs.png             ← Full 9-panel evaluation dashboard
    ├── model_comparison_table.png         ← Styled comparison table image
    ├── model_comparison_table.csv         ← Machine-readable results
    └── lstm_training_curves.png           ← BiLSTM training curves
```

---

## Task 1 – Dataset Understanding

| Property | Value |
|---|---|
| Records | 1,500 |
| Columns | ticket_id, channel, customer_message, sentiment_label, word_count, urgent_flag |
| Target | `sentiment_label` — negative / neutral / positive |
| Missing values | None |
| Class balance | negative=497 (33.1%) · neutral=524 (34.9%) · positive=479 (31.9%) — balanced |
| Avg word count | 12.72 words (range: 7–26) |
| Channels | email, social, phone, chat, app |

**Sample messages**:
- `[NEGATIVE]`: "I have raised multiple tickets but nobody has fixed the problem."
- `[NEUTRAL]`: "Can you confirm whether my ticket has been assigned?"
- `[POSITIVE]`: "The app experience is smooth and reliable. I appreciate the quick response."

---

## Task 2 – Text Preprocessing

| Step | Operation | Example |
|---|---|---|
| 1 | **Lowercase** | `"I Need Info"` → `"i need info"` |
| 2 | **Remove URLs** | `http://link.com` → `""` |
| 3 | **Remove standalone numbers** | ticket numbers like `78732` → removed |
| 4 | **Remove special characters** | `"Fast & reliable!"` → `"fast reliable"` |
| 5 | **Tokenise** | split on whitespace |
| 6 | **Remove stopwords** | `"i", "the", "and", "please"` → removed |
| 7 | **Padding/Truncation** | sequences padded to length 30 for LSTM |

**Result**: Average token count reduced from 12.72 → 6.0 (50% reduction, noise removed)

---

## Task 3 – Text Vectorization

### Why text must be converted to vectors

Machine learning models perform mathematical operations (dot products, matrix multiplications, gradient descent). Text is symbolic — the word `"angry"` has no numerical meaning on its own. Vectorisation transforms words into numbers that encode frequency, importance, or semantic relationships, making text compatible with optimisation algorithms.

### Methods used

| Method | Shape (train) | What it captures | Limitation |
|---|---|---|---|
| **Bag of Words** (1+2-gram) | 1200 × 385 | Word/phrase frequency | Loses word order; sparse |
| **TF-IDF** (1+2-gram) | 1200 × 385 | Word importance (penalises common words) | Loses word order; sparse |
| **Tokenizer Sequences** (for LSTM) | 1200 × 30 | Word order preserved | Requires more data; slower |

**Key insight**: BoW treats "not good" and "good not" identically. TF-IDF improves weighting but still loses order. Only sequence representations (embeddings + LSTM/Transformer) capture the **order and context** that determine true sentiment.

---

## Task 4 – Baseline Model Results

Six classical ML classifiers trained with two vectorisation methods:

| Vectorizer | Model | Test Accuracy | Macro F1 |
|---|---|---|---|
| BoW | Logistic Regression | **1.0000** | **1.0000** |
| BoW | Naive Bayes | 1.0000 | 1.0000 |
| BoW | Linear SVM | 1.0000 | 1.0000 |
| TF-IDF | Logistic Regression | 1.0000 | 1.0000 |
| TF-IDF | Naive Bayes | 1.0000 | 1.0000 |
| TF-IDF | Linear SVM | 1.0000 | 1.0000 |

### ⚠️ Why 100%? — A Critical Dataset Finding

This dataset is **template-generated**. All 1,500 messages are constructed from a vocabulary of only **146 unique tokens** after cleaning. Each sentiment class uses a non-overlapping vocabulary cluster:

- **Positive**: `appreciate`, `smooth`, `reliable`, `excellent`, `satisfied`
- **Negative**: `delay`, `issue`, `problem`, `wrong`, `charged`, `broken`
- **Neutral**: `information`, `confirm`, `check`, `scheduled`, `status`

A single token is therefore sufficient to classify any message perfectly. In production, customer messages contain ambiguity, sarcasm, and mixed sentiment — this is precisely why sequence models and transformers exist.

---

## Task 5 – Sequence Model: Bidirectional LSTM

### Architecture

```
Input Sequence (length=30 padded token IDs)
    ↓
Embedding(vocab=5000, dims=64)        — learns dense word representations
    ↓
Bidirectional LSTM(64 units)          — reads sequence forward AND backward
  → return_sequences=True             — outputs hidden state per timestep
    ↓
Dropout(0.4)
    ↓
Bidirectional LSTM(32 units)          — condenses into final context vector
    ↓
Dropout(0.4)
    ↓
Dense(64, ReLU)                       — non-linear classifier head
    ↓
Dropout(0.4)
    ↓
Dense(3, Softmax)                     — P(negative), P(neutral), P(positive)
```

| Component | Configuration | Rationale |
|---|---|---|
| Loss | Sparse Categorical Cross-Entropy | Multi-class (3 labels) |
| Optimizer | Adam (lr=0.001) | Adaptive gradients; stable for RNNs |
| Embedding | 64-dim trainable | Learned from data, no pre-trained weights needed |
| Bidirectional | ✅ | Forward context + backward context for richer representation |
| Early Stopping | patience=5 | Prevents overfitting on small vocabulary |

### Experiment Comparison

| Model | LSTM Units | Dropout | LR | Test Acc | Macro F1 |
|---|---|---|---|---|---|
| **BiLSTM Baseline** | 64 | 0.4 | 0.001 | **82.33%** | **0.805** |
| Smaller LSTM | 32 | 0.3 | 0.001 | 64.67% | 0.578 |
| Larger + High Dropout | 128 | 0.5 | 0.0005 | 35.00% | 0.173 |

**Why LSTM underperforms classical ML here**: With only 146 unique tokens, the Embedding layer provides no advantage over BoW — there is insufficient vocabulary to learn meaningful dense representations. On a real-world dataset with thousands of diverse tokens, LSTMs would significantly outperform bag-of-words approaches.

---

## Task 6 – Attention and Transformer Reflection

### Why RNNs struggle with long-term dependencies

RNNs maintain a single **hidden state vector** passed from timestep to timestep. To connect the 1st word to the 50th word, the gradient must flow through 49 matrix multiplications during backpropagation. At each step, the gradient is multiplied by the weight matrix — if the matrix has singular values < 1, the gradient vanishes exponentially; if > 1, it explodes. This means RNNs effectively cannot remember information from more than ~10–20 steps ago.

**Concrete example**: In `"The customer, who had been waiting for three weeks despite multiple follow-up calls and escalation tickets that were ignored, was **angry**."` — an RNN would likely forget `"The customer"` by the time it reaches `"angry"`.

---

### How LSTMs help with memory

LSTMs introduce a separate **cell state** (c_t) — a "conveyor belt" of long-term memory — controlled by three learnable gates:

| Gate | Formula (simplified) | Function |
|---|---|---|
| **Forget gate** (f) | σ(W_f · [h_{t-1}, x_t]) | Decides what % of cell memory to erase |
| **Input gate** (i) | σ(W_i · [h_{t-1}, x_t]) | Decides what new info to write to cell |
| **Output gate** (o) | σ(W_o · [h_{t-1}, x_t]) | Controls what the cell exposes as hidden state |

The cell state update `c_t = f ⊙ c_{t-1} + i ⊙ g̃` allows the gradient to flow through time with **constant error carousels** — the forget gate can be set to 1.0, letting information pass unchanged across hundreds of steps. This solves the vanishing gradient problem.

---

### What attention solves in sequence-to-sequence tasks

In encoder-decoder architectures (e.g. machine translation), the encoder must compress an entire input sentence — regardless of length — into a **single fixed-size vector**. For long sentences this is a severe bottleneck: nuanced details in the middle of the source sentence are lost before the decoder ever starts generating.

**Attention** removes this bottleneck:
1. The encoder produces a hidden state for **every input token**: h_1, h_2, ..., h_n
2. At each decoder step, an alignment score e_{ti} = score(s_t, h_i) is computed for every encoder state
3. Scores are normalised to attention weights: α_{ti} = softmax(e_{ti})
4. A context vector c_t = Σ α_{ti} · h_i is computed — a weighted combination of all encoder states

This lets the model focus on the relevant source words for each output word. For example, translating "bank" depends on whether the surrounding words say "river" or "money" — attention can look directly at those words.

---

### Why Transformers are important in modern NLP and Generative AI

Transformers (Vaswani et al., 2017 — "Attention is All You Need") replace recurrence entirely with **multi-head self-attention**, yielding three decisive advantages:

| Advantage | RNN/LSTM | Transformer |
|---|---|---|
| **Parallelisation** | Sequential — step n needs step n-1 | Fully parallel — all tokens computed simultaneously |
| **Long-range connections** | O(n) steps between distant tokens | O(1) — any token attends to any other directly |
| **Scalability** | Training bottlenecked by sequential computation | Trains efficiently on billions of parameters |

**Self-attention** allows each word to attend to every other word in the same sentence simultaneously, building rich contextual representations without any recurrence.

**In Generative AI**: The GPT family, Llama, Gemini, Claude, and all modern large language models are decoder-only Transformers trained with next-token prediction. Key innovations built on top:
- **Positional encodings** — inject word order since attention is permutation-invariant
- **Multi-head attention** — learn multiple attention patterns simultaneously (syntax, coreference, semantics)
- **Layer normalisation + residual connections** — enable stable training at scale
- **RLHF** — aligns model outputs with human preferences post-training

Transformers have become the universal backbone for NLP, computer vision (ViT), audio (Whisper), and multimodal AI (CLIP, Gemini), making them the most important architectural advance in AI since backpropagation.

---

## How to Run

```bash
git clone https://github.com/<your-username>/part-3-nlp-sequence-modeling
cd part-3-nlp-sequence-modeling
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

All outputs saved to `results/` automatically.

---

## Requirements

```
tensorflow>=2.10
scikit-learn>=1.2
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
jupyter>=1.0
```