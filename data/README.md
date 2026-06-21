# Data

Raw dataset files are **not committed** to this repository due to file size
and licence restrictions. Follow the instructions below to set them up locally.

## Directory layout expected by the notebook

```
data/
└── mydata/
    ├── user_events.txt              # LFM-1b listening events (1.1 GB)
    ├── low_main_users.txt           # Kowald mainstreaminess groups
    ├── medium_main_users.txt
    ├── high_main_users.txt
    ├── data2/
    │   ├── LFM-1b_artists.txt       # Artist ID → name mapping
    │   └── LFM-1b_users.txt         # User demographics
    ├── LFM-1b_UGP/
    │   ├── genres_allmusic.txt      # Genre name list (one per line)
    │   └── LFM-1b_artist_genres_allmusic.txt
    └── data3/
        └── xite_msd.parquet         # XITE Million Sessions dataset
```

## Download links

- **LFM-1b**: http://www.cp.jku.at/datasets/LFM-1b/
- **LFM-1b UGP (genre tags)**: included in the same download package
- **XITE**: contact the dataset authors or use the version provided in the course

## Synthetic fallback

If `user_events.txt` is absent, the notebook automatically generates synthetic
listening events that replicate the power-law popularity distribution.
All bias metrics and model training work on the synthetic data, though
the specific numbers will differ from the paper results.
