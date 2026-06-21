# Measuring & Mitigating Popularity Bias in Music Recommender Systems

> **WU Vienna · Data Science & Artificial Intelligence · 2026**
> Ivanna Levchenko (h12200708) · Oleksandr Ursol (h12438168)

---

## Overview

Popularity bias in music recommender systems is the tendency of algorithms to over-recommend already-mainstream artists while systematically suppressing niche and long-tail musicians. This project:

1. **Measures** bias on two real-world datasets (LFM-1b, XITE) across four dimensions — artist-side, user-side, genre-side, and demographic-side.
2. **Amplification analysis** — shows that all three baseline recommenders (Popularity, ItemKNN, ALS) push ARP above the raw-data baseline.
3. **Mitigates** bias via two post-processing re-ranking strategies (xQuAD, FairRerank) applied to the ALS model.
4. **Quantifies trade-offs** — a Pareto analysis of NDCG@10 (accuracy) vs ACLT (long-tail fairness).
5. **Ships a reusable toolkit** — `popbias` — for measuring popularity bias in any RecSys output.

---

## Key Results

| Model | ARP ↑ (bias) | ACLT ↑ (fairness) | NDCG@10 |
|---|---|---|---|
| Raw data baseline | 4,996 | — | — |
| PopularityRec | 67,739 | 0.000 | 0.108 |
| ItemKNN | 303 | 0.578 | 0.015 |
| ALS | 15,215 | 0.000 | 0.238 |
| **ALS + xQuAD λ=0.45** | — | **0.323** | 0.190 |
| **ALS + FairRerank min=6** | — | **0.600** | 0.148 |

**Genre bias** (LFM-1b): top 4 genres (rock, alternative, electronic, pop) capture ~64% of plays; Genre Gini = 0.5949, reproduced on XITE (0.5173).

---

## Repository Structure

```
popularity-bias-music-recsys/
│
├── popbias/                   # Reusable Python toolkit (pip-installable)
│   ├── __init__.py
│   ├── metrics.py             # gini, arp, aclt, catalog_coverage, rec_metrics
│   └── mitigation.py          # xquad_rerank, fair_rerank
│
├── notebooks/
│   └── popularity_bias_FULL_v5.ipynb   # Full end-to-end project (Weeks 1–5)
│
├── data/
│   └── processed/             # Lightweight outputs saved by the notebook
│       ├── artist_popularity.csv
│       ├── genre_plays.csv
│       ├── user_bias_metrics.csv
│       └── ...
│
├── results/                   # Summary tables (bias metrics, Pareto points)
│   ├── bias_on_recommendations.csv
│   └── mitigation_results.csv
│
├── docs/                      # Presentation slides
│   └── popularity_bias_final.pptx
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/popularity-bias-music-recsys.git
cd popularity-bias-music-recsys
pip install -r requirements.txt
pip install -e .
```

### 2. Use the `popbias` toolkit

```python
from popbias import gini, popularity_groups, arp, aclt, xquad_rerank

# --- Measure bias in raw consumption data ---
item_plays = {"artist_A": 50000, "artist_B": 200, "artist_C": 50}

groups = popularity_groups(item_plays)
# {"artist_A": "head", "artist_B": "mid", "artist_C": "tail"}

print("Gini:", gini(list(item_plays.values())))  # 0.71

# --- Measure bias in recommendation lists ---
tail_set = {k for k, v in groups.items() if v == "tail"}
rec_lists = [["artist_A", "artist_A", "artist_B"], ["artist_A", "artist_C"]]

print("ARP:",  arp(rec_lists, item_plays))   # high = biased
print("ACLT:", aclt(rec_lists, tail_set))    # low  = biased

# --- Mitigate with xQuAD re-ranking ---
cand_ids    = ["artist_A", "artist_B", "artist_C"]
cand_scores = [0.95, 0.80, 0.60]

reranked = xquad_rerank(cand_ids, cand_scores, lam=0.45, tail_set=tail_set)
print("Re-ranked:", reranked)
```

### 3. Run the full notebook

```bash
cd notebooks
jupyter notebook popularity_bias_FULL_v5.ipynb
```

> **Data note:** Raw LFM-1b and XITE files are not included (licence restrictions and file size). See [Data Setup](#data-setup) below.

---

## Data Setup

### LFM-1b (main dataset)
- Download from: http://www.cp.jku.at/datasets/LFM-1b/
- Place `user_events.txt`, `LFM-1b_artists.txt`, `LFM-1b_users.txt` in `data/mydata/`
- Place genre files in `data/mydata/LFM-1b_UGP/`
- Mainstreaminess group files (`low_main_users.txt`, etc.) from the same source

### XITE (cross-dataset validation)
- XITE Million Sessions dataset (music video platform, 2023-24)
- Place `xite_msd.parquet` in `data/mydata/data3/`

If the data files are absent the notebook falls back to a synthetic event generator that replicates the power-law distribution for development purposes.

---

## Methodology

### Pipeline

```
Data Collection → Pre-processing → Model Training → Bias Measurement → Mitigation & Eval
   LFM-1b            Filter           PopularityRec      ARP                xQuAD
   XITE              Normalise         ItemKNN (K=50)     ACLT               FairRerank
                     80/20 split       ALS (f=64)         Gini               NDCG / F-score
```

### Bias Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **ARP** | (1/\|U\|) Σ_u (1/\|L_u\|) Σ_i φ(i) | Mean popularity of recommended items. Higher = more mainstream bias. |
| **ACLT** | (1/\|U\|) Σ_u (Σ_i 𝟙[i∈Γ]) / \|Γ\| | Fraction of long-tail items covered per user. Lower = more bias. |
| **Gini** | Σ_i Σ_j \|x_i−x_j\| / (2n²·x̄) | Inequality of item exposure. 0 = equal; 1 = monopoly. |

### Mitigation

**xQuAD re-ranking** (user-side):
```
val = (1 - λ) · relevance + λ · tail_bonus
```
Sweep λ ∈ [0, 1] to trace the accuracy–fairness Pareto frontier.

**FairRerank** (provider/artist-side):
Guarantee `min_tail` long-tail artists in every Top-N list.

---

## Limitations

- **Implicit feedback only** — play counts ≠ explicit ratings; a song left on repeat inflates the count.
- **LFM-1b is European-skewed** — findings may not generalise to other cultural contexts.
- **No hyperparameter tuning** — ALS factors, K, and λ were not cross-validated.
- **Adversarial training not implemented** — only post-processing mitigation is included.
- **Cold-start excluded** — all evaluation users have training history.
- **XITE uses sessions as pseudo-users** — not directly comparable to LFM-1b user-side metrics.

---

## References

- Abdollahpouri et al. (2019). *Managing Popularity Bias in Recommender Systems*. FLAIRS 2019.
- Celma, O. (2010). *Music Recommendation and Discovery*. Springer.
- Ferraro et al. (2021). *Break the Loop: Gender Imbalance in Music Recommenders*. ACM UMAP 2021.
- Schedl, M. (2016). *The LFM-1b Dataset for Music Retrieval and Recommendation*. ACM ICMR 2016.
- Clarke et al. (2008). *Novelty and Diversity in Information Retrieval Evaluation*. SIGIR 2008. (xQuAD)

---

## License

MIT — see [LICENSE](LICENSE).
The LFM-1b and XITE datasets are subject to their own licences; see the respective dataset pages.
