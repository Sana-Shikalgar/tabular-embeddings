## 01 Overview & Cleaning: TMDB -- summary (2026-08-10 21:34)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 1,253,097
- **TMDB columns (raw -> cleaned)**: 24 -> 18
- **Raw memory footprint (deep)**: 734.2 MB
- **Dropped: status != 'Released'**: 54,181 rows
- **Dropped: adult == True**: 143,475 rows
- **Recoded budget/revenue zeros as NaN**: 93.8% / 98.1% of rows
- **Recoded runtime zeros as NaN**: 30.5% of rows
- **Recoded negative runtime/revenue as NaN**: 0.0001% / 0.0001% of rows
- **Dropped: duplicate id rows**: 1,006 (kept most complete row per id)
- **Dropped: duplicate imdb_id rows**: 3,748 (kept most complete row per imdb_id, tie-broken by lowest id)
- **Dropped: missing title rows**: 13
- **Dropped: inconsistent vote_average/vote_count rows**: 1,309
- **Dropped column: popularity**: redundant with vote_average/vote_count
- **Final missingness (top 3 by %)**: {'revenue': 98.09, 'budget': 93.86, 'tagline': 85.99}
- **vote_count >= 1 / >= 5 / >= 10**: 337,742 / 125,066 / 78,659 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\preprocessed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

