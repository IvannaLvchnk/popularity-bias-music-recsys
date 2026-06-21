"""
popbias — Popularity Bias Measurement Toolkit for Music Recommender Systems
===========================================================================
Authors : Ivanna Levchenko (h12200708) · Oleksandr Ursol (h12438168)
Course  : Data Science & Artificial Intelligence, WU Vienna, 2026

Provides dataset-agnostic functions for measuring and mitigating popularity
bias in any recommender-system output. Implements metrics from:
  - Abdollahpouri et al. (2019) FLAIRS — ARP, ACLT
  - Celma (2010) / Schedl (2016) — Gini, long-tail partitioning
"""

from .metrics import gini, popularity_groups, arp, aclt, catalog_coverage, rec_metrics
from .mitigation import xquad_rerank, fair_rerank

__all__ = [
    "gini",
    "popularity_groups",
    "arp",
    "aclt",
    "catalog_coverage",
    "rec_metrics",
    "xquad_rerank",
    "fair_rerank",
]

__version__ = "1.0.0"
