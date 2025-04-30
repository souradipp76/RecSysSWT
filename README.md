# Recommender-Systems Experiments &nbsp;<!-- README.md -->

A small suite of **PyTorch** scripts for prototyping sequence-based recommender-system models with different sampling and embedding strategies.

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `sliding.py` | Baseline training + evaluation on e-commerce event logs | Transformer encoder, classic `nn.Embedding`, two dataset sampling modes (control & sliding)  |
| `hashsliding.py` | **Hashed-ID variant** with *K-Shift* compressed embeddings | xxHash item hashing, `KShiftEmbedding`, frequency-based item filtering  |
| `inference.py` | Lightweight **checkpoint evaluation** script | Loads saved weights (`model.pth`), computes loss, perplexity, MRR, MAP, Recall@K  |

---

## 1. Quick Start

```bash
# 1. Create environment
conda create -n recsys python=3.10
conda activate recsys
pip install torch pandas numpy tqdm xxhash

# 2. Place your CSV under data/
#    Expected columns vary per script (see §3).

# 3. Train baseline
python sliding.py          # default = control mode
# 4. Evaluate / extra epoch
python inference.py        # uses saved model.pth
# 5. Train hashed variant
python hashsliding.py      # default = control mode
```

> **GPU:** Each script auto-detects CUDA and falls back to CPU.

---

## 2. Repository Layout

```
.
├── sliding.py          # baseline train + eval
├── inference.py        # checkpoint eval / inference
├── hashsliding.py      # hashed-ID, K-Shift variant
└── data/
    ├── events.csv      # sample file for baseline
    └── UserBehavior.csv# sample file for hashed variant
```

Outputs
```
model.pth               # saved every epoch
```

---

## 3. Dataset Format

### 3.1 Baseline (`sliding.py`, `inference.py`)
CSV columns (no header in file required, *header provided in script*):

| visitorid | itemid | event | timestamp | transactionid |
|-----------|--------|-------|-----------|---------------|

`event` must be one of **view / addtocart / transaction** (mapped to `0/1/2`) .

### 3.2 Hashed variant (`hashsliding.py`)
CSV columns **without** header:

| user_id | item_id | category_id | interaction_type | timestamp |
|---------|---------|-------------|------------------|-----------|

`interaction_type` must be **pv / buy / cart / fav** (mapped to `0‥3`) .

Low-frequency items ( < 10 interactions) are dropped before training. 

---

## 4. Sampling Modes

| Mode | Description | Implemented in |
|------|-------------|----------------|
| **control** | Last `window_size` events of each user | all scripts |
| **sliding** | Sliding window over each user history with stride `sliding_stride` | all scripts |
| **mixed**   | First *X* epochs control, remaining epochs sliding (see `train_model_mixed`) | `sliding.py`, `hashsliding.py`  |

Key CLI parameters (all have sensible defaults):

```python
window_size   = 100       # max tokens fed into Transformer
max_history   = 500/1000  # truncate long user histories (sliding modes)
sliding_stride= 1         # hop length
batch_size    = 16 / 32   # GPUs love powers of two
num_epochs    = 5 / 10
lr            = 1e-3
```

---

## 5. Model Architecture

### 5.1 `RecSysFM` (all scripts)
```
item/interaction/position Embeddings
           ↓ (sum)
TransformerEncoder (N layers, H heads)
           ↓
Linear projection → item vocabulary
```
* **Causal mask** (`triu(-inf)`) enforces autoregressive flow .
* Loss: **Cross-Entropy** on next-item prediction.

### 5.2 K-Shift Embedding (hashed variant)
`hashsliding.py` replaces the item `nn.Embedding` with a **KShiftEmbedding** layer:

1. xxHash64 each raw `item_id` with per-field seed → 64-bit int  
2. Bit-shift the hash up to `num_shifts` times, look up **shared** embedding rows, and sum  
3. Optionally L2-normalize output 

This yields a tiny table (≈ 1.15 × `num_items`) yet retains capacity via hashing and rotation.

---

## 6. Training & Evaluation Workflow

1. **Dataset** → `InteractionDataset`  
   *Generates (input_items, target_items, input_types) tensors; dynamic padding in `pad_collate_fn`.*

2. **Training** (`train_model` / `train_model_mixed`)  
   *Saves `model.pth` after every epoch.*

3. **Evaluation** (`evaluate_model`)  
   *Metrics*  
   * Loss & Perplexity  
   * **MRR**, **MAP** (same as MRR when one ground-truth per step), **Recall@K** 

---

## 7. Reproducibility Tips

* Fix seeds in NumPy / Torch for deterministic runs.
* Pin versions (e.g., `torch==2.2.*`, `pandas==2.*`) in `environment.yml`.
* Keep the same `window_size` during training **and** inference.

---

## 8. Extending the Code

* **Different Events**: edit the `*_MAPPING` dicts.  
* **Bigger Models**: increase `emb_dim`, `n_layers`, `n_heads`.  
* **Alternate Losses**: swap `nn.CrossEntropyLoss` for sampled softmax, BPR, etc.  
* **Cold-start**: integrate side features into `hash_feature`.

---

## 9. Citation

If you use this codebase in academic work, please cite:

```
@misc{recsysfm2025,
  title   = {RecSysFM Experiments},
  author  = {Anonymous},
  year    = {2025},
  howpublished = {\url{https://github.com/your-fork/recsysfm}}
}
```

---

## 10. License

MIT License – see `LICENSE` file.

Enjoy experimenting! Feel free to open issues or pull requests.
