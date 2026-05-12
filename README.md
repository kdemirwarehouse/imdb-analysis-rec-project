# 🎬 Movie Recommendation System

A production-grade, end-to-end movie analytics & recommendation platform built
on **IMDB**, **TMDB**, and **MovieLens 25M** data. Includes EDA dashboards,
country-level analysis, and three recommender families (content-based,
collaborative filtering, hybrid) — all surfaced through a Streamlit app.

> **Status:** Step 1 / 10 complete (environment scaffolding).

---

## 📁 Project Structure

```
movie_recommendation_project/
├── config.py                  # Central configuration (paths, hyperparams, logging)
├── requirements.txt           # Pinned dependencies
├── .env.example               # Template for the real .env (TMDB API key)
├── .gitignore
│
├── data/
│   ├── raw/                   # IMDB .tsv.gz, MovieLens .zip
│   ├── processed/             # movies_clean.csv, country_stats.csv
│   └── external/              # TMDB cache
│
├── notebooks/                 # 01..08 — exploratory + reproducible workflow
│
├── src/
│   ├── data/                  # downloader.py, tmdb_fetcher.py, preprocessor.py
│   ├── eda/                   # visualizations.py, country_analysis.py
│   └── recommender/           # content_based.py, collaborative.py, hybrid.py
│
├── app/                       # Streamlit dashboard
│   ├── streamlit_app.py
│   ├── pages/
│   └── components/
│
├── models/                    # Trained pickles (TF-IDF, SVD, ...)
└── reports/figures/           # Generated charts
```

---

## ⚙️ Quick Start

```bash
# 1. Clone and enter
git clone <repo>
cd movie_recommendation_project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# -> open .env and paste your TMDB_API_KEY

# 5. Smoke-test the configuration
python config.py
```

Expected output:
```
... | INFO     | config:N | Project root: /.../movie_recommendation_project
... | INFO     | config:N | Random state: 42
... | INFO     | config:N | TMDB configured: True
... | INFO     | config:N | IMDB files: title.basics.tsv.gz, ...
```

---

## 🧪 Reproducibility

* `RANDOM_STATE = 42` fixed in `config.py` and propagated to numpy / sklearn / surprise.
* Dependency versions pinned with `~=` (compatible release).
* All paths are project-relative (`pathlib.Path`).

---


## 📚 Data Sources

| Source | License | Used for |
|---|---|---|
| [IMDB Datasets](https://datasets.imdbws.com/) | Personal & non-commercial | Titles, ratings, crew |
| [TMDB API](https://www.themoviedb.org/documentation/api) | Free w/ attribution | Country, budget, revenue, overview |
| [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) | GroupLens research | User–movie ratings |

> _This product uses the TMDB API but is not endorsed or certified by TMDB._
