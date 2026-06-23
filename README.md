# Measuring & Mitigating Popularity Bias in Music Recommender Systems

**WU Vienna · MSc Data Science & Artificial Intelligence · 2026**  
Ivanna Levchenko (h12200708) · Oleksandr Ursol (h12438168)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: LFM-1b](https://img.shields.io/badge/Dataset-LFM--1b-orange)](http://www.cp.jku.at/datasets/LFM-1b/)

---

## Motivation

Have you ever noticed that Spotify keeps recommending the same artists over and over? There is a reason for that — and it is not just your taste.

Music recommender systems suffer from **popularity bias**: algorithms trained on historical listening data learn that popular artists get played more, so they recommend popular artists even more, which makes them *even more* popular. This self-reinforcing feedback loop is sometimes called the **Matthew Effect** — *the rich get richer* (Merton, 1968).

The consequences are tangible. Niche and emerging artists lose visibility and revenue even when their music genuinely matches a listener's taste. Cultural diversity shrinks as algorithmic monocultures push listeners toward the same handful of mainstream acts. And listener trust erodes when a "personalised" playlist feels indistinguishable from a popularity chart.

Despite growing awareness of the problem, most existing fairness toolkits — Fairlearn, AI Fairness 360 — focus exclusively on demographic bias and offer no support for item-level popularity bias in recommender systems. This project fills that gap by providing both a rigorous empirical study and a reusable measurement toolkit.

---

## Research Questions

**RQ1 — Measurement.** Which metrics (ARP, ACLT, Gini) best capture the extent of popularity bias across different recommender algorithms — Popularity, ItemKNN, and ALS?

**RQ2 — Mitigation.** Do post-processing re-ranking strategies (xQuAD, FairRerank) reduce popularity bias while preserving recommendation quality?

**RQ3 — Trade-offs.** How do mitigation strategies affect long-tail artist exposure, catalog coverage, and NDCG accuracy — and where is the Pareto-optimal compromise?

---

## Key Findings

### The raw data is already severely skewed

Before any algorithm touches it, the consumption data exhibits extreme concentration:

- The top 20% of artists capture **96% of all plays** on LFM-1b (Gini = 0.93)
- Reproduced on XITE: top 20% → 97.5% of plays, Gini = 0.93
- User groups differ significantly: high-mainstreaminess users have ARP = 15,965 vs 6,336 for low-mainstreaminess users (Mann-Whitney p < 0.001)

### All three recommenders amplify bias above the raw-data level

| Model | ARP (bias, higher is worse) | ACLT (fairness, higher is better) | NDCG@10 |
|:---|---:|---:|---:|
| Raw data baseline | 4,996 | — | — |
| PopularityRec | 67,739 | 0.000 | 0.108 |
| ItemKNN (K=50) | 303 | 0.578 | 0.015 |
| ALS (factors=64) | 15,215 | 0.000 | 0.238 |

PopularityRec represents the extreme case — it recommends the same 60 artists to every single user, achieving a catalog coverage of just 0.02%. ALS achieves the best accuracy but still pushes ARP to three times the raw-data baseline.

### Mitigation works, but involves a genuine trade-off

| Method | NDCG@10 | ACLT | Catalog Coverage |
|:---|---:|---:|---:|
| ALS (no mitigation) | 0.238 | 0.000 | 1.1% |
| xQuAD λ=0.45 | 0.190 | 0.323 | 2.9% |
| xQuAD λ=0.50 | 0.001 | 1.000 | 4.0% |
| FairRerank min=4 | 0.184 | 0.400 | 3.1% |
| FairRerank min=6 | 0.148 | 0.600 | 3.6% |

**λ=0.45 is the empirically identified sweet spot** — it recovers 32% long-tail coverage for a cost of approximately 0.04 NDCG points. Beyond λ=0.50, accuracy collapses entirely, illustrating the hard boundary of the fairness-accuracy trade-off.

### Genre-side bias is structurally persistent

We extended the analysis to genre level — something rarely done in the popularity bias literature:

- The top 4 genres (rock, alternative, electronic, pop) account for approximately 64% of all LFM-1b plays
- Genre Gini = 0.59 on LFM-1b and 0.52 on XITE — moderate but stable across datasets
- The dominant genre differs by platform: rock/alternative on LFM-1b (scrobbling-era), Country/Pop on XITE (streaming-era), reflecting different user bases rather than universal preferences

---

## Repository Structure

```
popularity-bias-music-recsys/
│
├── notebooks/
│   └── popularity_bias_FULL_v5.ipynb   # End-to-end analysis (Weeks 1-5)
│
├── popbias/                             # Reusable Python toolkit
│   ├── __init__.py
│   ├── metrics.py                       # gini, arp, aclt, catalog_coverage, rec_metrics
│   └── mitigation.py                    # xquad_rerank, fair_rerank
│
├── data/
│   └── README.md                        # Dataset download instructions
│
├── results/
│   ├── bias_on_recommendations.csv      # Baseline model bias metrics
│   ├── mitigation_results.csv           # Full lambda-sweep Pareto data
│   └── raw_bias_summary.csv             # LFM-1b and XITE summary statistics
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Architecture

The project follows a five-stage pipeline:

```
Data Collection --> Pre-processing --> Model Training --> Bias Measurement --> Mitigation
   LFM-1b            Filter             PopularityRec      ARP                xQuAD
   XITE               Normalise          ItemKNN (K=50)     ACLT               FairRerank
                      80/20 split        ALS (f=64)         Gini               NDCG / Recall
```

### Bias metrics

**ARP (Average Recommendation Popularity)** measures the mean global play count of recommended items across all users. A high ARP indicates the recommender is steering users toward already-mainstream content.

**ACLT (Average Coverage of Long-Tail)** measures what fraction of each user's recommendations come from the long-tail item set (bottom 80% by popularity). A low ACLT indicates niche artists are being systematically excluded.

**Gini coefficient** captures the inequality of item exposure across all recommendation lists. A value of 0 means every item is recommended equally often; a value of 1 means a single item receives all recommendations.

### Mitigation strategies

**xQuAD re-ranking (user-side).** At each greedy step, candidates are scored by:

```
val(i) = (1 - lambda) * relevance(i) + lambda * tail_bonus(i)
```

where `tail_bonus = 1` if the artist is in the long-tail, else `0`. Sweeping lambda across [0, 1] traces the full accuracy-fairness Pareto frontier.

**FairRerank (provider-side).** Guarantees that at least `min_tail` long-tail artists appear in every Top-N recommendation list, providing a hard fairness floor for niche musicians regardless of their predicted relevance score.

---

## Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab
- Approximately 2 GB of disk space for the LFM-1b dataset

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/IvannaLvchnk/popularity-bias-music-recsys.git
cd popularity-bias-music-recsys
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Download the datasets

**LFM-1b** (1B+ listening events, 352K artists, official mainstreaminess labels):
```
http://www.cp.jku.at/datasets/LFM-1b/
```
Place `user_events.txt`, `LFM-1b_artists.txt`, `LFM-1b_users.txt`, and the UGP genre files under `data/mydata/` as described in [`data/README.md`](data/README.md).

**XITE** (23M music video sessions, cross-dataset validation):  
Place `xite_msd.parquet` under `data/mydata/data3/`.

If neither dataset is present, the notebook automatically falls back to a synthetic power-law event generator so the code and metrics can be explored without downloading anything.

### 4. Run the notebook

```bash
jupyter notebook notebooks/popularity_bias_FULL_v5.ipynb
```

---

## Using the popbias Toolkit

The `popbias` module is dataset-agnostic and can be applied to any recommender system output.

```python
from popbias import gini, popularity_groups, arp, aclt
from popbias import xquad_rerank, fair_rerank

# Characterise your item catalog
item_plays = {"radiohead": 50000, "arctic_monkeys": 8000, "bedroom_pop_band": 120}
groups     = popularity_groups(item_plays)
tail_set   = {k for k, v in groups.items() if v == "tail"}

print(f"Gini: {gini(list(item_plays.values())):.3f}")

# Measure bias in recommendation lists
rec_lists = [["radiohead", "arctic_monkeys"], ["radiohead", "bedroom_pop_band"]]
print(f"ARP  (higher = more biased): {arp(rec_lists, item_plays):.0f}")
print(f"ACLT (lower  = more biased): {aclt(rec_lists, tail_set):.3f}")

# Mitigate with xQuAD re-ranking
cand_ids    = ["radiohead", "arctic_monkeys", "bedroom_pop_band"]
cand_scores = [0.95, 0.80, 0.60]
reranked = xquad_rerank(cand_ids, cand_scores, lam=0.45, tail_set=tail_set)

# Guarantee niche artists with FairRerank
fair = fair_rerank(cand_ids, cand_scores, min_tail=1, tail_set=tail_set)
```

---

## Limitations

We want to be transparent about the boundaries of this work.

**Implicit feedback only.** Play counts are not explicit ratings. A song left on repeat, or music playing in the background, inflates the count — meaning all bias metrics are computed on an inherently noisy signal.

**LFM-1b is culturally skewed.** The Last.fm user base skews heavily toward European (particularly German-speaking) listeners. Genre distributions and artist popularity findings may not generalise to Latin American, East Asian, or African music ecosystems.

**No hyperparameter tuning.** ALS (factors=64, regularisation=0.05) and ItemKNN (K=50) were run with sensible defaults rather than cross-validated. Optimal settings could shift the accuracy-fairness Pareto frontier.

**Adversarial training not implemented.** The original proposal included in-processing debiasing as a third mitigation strategy. Only post-processing (xQuAD, FairRerank) was completed in the project timeframe; in-processing remains future work.

**Cold-start users excluded.** All evaluation users have a training history. Cold-start users — arguably those who would benefit most from diversity-promoting recommendations — are entirely outside the scope of this study.

**XITE uses sessions as pseudo-users.** Session IDs are anonymous and short-lived; they do not represent stable long-term taste profiles, which limits the comparability of XITE-side bias metrics with LFM-1b user-side results.

---

## References

1. Abdollahpouri, H., Mansoury, M., Burke, R., & Mobasher, B. (2019). *The Unfairness of Popularity Bias in Recommendation*. FLAIRS 2019.
2. Celma, O. (2010). *Music Recommendation and Discovery: The Long Tail, Long Fail, and Long Play in the Digital Music Space*. Springer.
3. Clarke, C. L. A., Kolla, M., Cormack, G. V., et al. (2008). *Novelty and Diversity in Information Retrieval Evaluation*. SIGIR 2008.
4. Ferraro, A., Schedl, M., & Serra, X. (2021). *Break the Loop: Gender Imbalance in Music Recommenders*. ACM UMAP 2021.
5. Kowald, D., Schedl, M., & Lex, E. (2020). *The Unfairness of Popularity Bias in Music Recommendation: A Reproducibility Study*. ECIR 2020.
6. Merton, R. K. (1968). *The Matthew Effect in Science*. Science, 159(3810), 56–63.
7. Schedl, M. (2016). *The LFM-1b Dataset for Music Retrieval and Recommendation*. ACM ICMR 2016.

---

## Authors

**Ivanna Levchenko** · h12200708@s.wu.ac.at  
**Oleksandr Ursol** · h12438168@s.wu.ac.at  
MSc Data Science & Artificial Intelligence, WU Vienna, 2026

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.  
The LFM-1b and XITE datasets are subject to their own licences; please consult the respective dataset pages before use.
