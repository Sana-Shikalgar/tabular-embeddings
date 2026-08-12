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

## 01 Overview & Cleaning: TMDB -- summary (2026-08-10 22:06)

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

## 01 Overview & Cleaning: TMDB -- summary (2026-08-10 22:08)

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

## 01 Overview & Cleaning: TMDB -- summary (2026-08-11 00:42)

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

## 01 Overview & Cleaning: TMDB -- summary (2026-08-11 01:04)

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

## 00 Characterization: TMDB — summary (2026-08-11 01:20)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 50}
- **TMDB overview: % > 512 tokens (sampled)**: 0.0%
- **TMDB overview: % flagged non-English (sampled)**: 5.0%
- **TMDB overview: % HTML-contaminated**: 0.01%
- **TMDB genres vocabulary size**: 19
- **TMDB keywords vocabulary size**: 68696
- **Multi-hot vs pooled-embedding recommendation**: genres: multi-hot feasible; keywords: use pooled SBERT-of-item-names if vocab_size > 5000 (see Section 5 warnings above)

Figures:
- `reports/figures/00_tmdb_vote_average_hist.png`
- `reports/figures/00_tmdb_boxplots.png`
- `reports/figures/00_tmdb_correlation_heatmap.png`
- `reports/figures/00_tmdb_overview_length_hist.png`

---

## 01 Overview & Cleaning: TMDB -- summary (2026-08-11 01:25)

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

## 01 Overview & Cleaning: TMDB -- summary (2026-08-11 01:45)

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
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

## 01 Overview & Cleaning: TMDB -- summary (2026-08-11 03:15)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 1,253,097
- **TMDB columns (raw -> cleaned)**: 24 -> 19
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
- **Final missingness (top 3 by %)**: {'revenue': 98.09, 'budget': 93.86, 'tagline': 85.99}
- **vote_count >= 1 / >= 5 / >= 10**: 337,742 / 125,066 / 78,659 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-11 18:13)

- **Base cleaned dataset (from 01)**: (1253097, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7535, 19), 'tmdb_budget_revenue_gt0': (11264, 20), 'tmdb_budget_revenue_gt5': (9186, 19), 'tmdb_budget_revenue_gt10': (8870, 19), 'tmdb_no_budget_revenue_gt0': (337742, 17), 'tmdb_no_budget_revenue_gt5': (110389, 18), 'tmdb_no_budget_revenue_gt10': (73898, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 261, 'tmdb_no_budget_revenue_gt5': 80}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (11003, 20), 'tmdb_no_budget_revenue_gt5': (110309, 18)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '76.4% / 95.6%', 'tmdb_no_budget_revenue_gt5': '39.1% / 97.9%'}
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '82.2% / 82.4%', 'tmdb_no_budget_revenue_gt5': '65.0% / 65.1%'}
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=716.11, p=2.378e-148', 'tmdb_no_budget_revenue_gt5': 'H=1025.86, p=4.57e-215'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the presence/equality flags used then dropped (see Sections 2, 3, 6, 7)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

## 01 Overview & Cleaning: TMDB -- summary (2026-08-12 08:36)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 871,113
- **TMDB columns (raw -> cleaned)**: 24 -> 19
- **Raw memory footprint (deep)**: 734.2 MB
- **Dropped: status != 'Released'**: 54,181 rows
- **Dropped: adult == True**: 527,840 rows
- **Recoded budget/revenue zeros as NaN**: 93.8% / 98.1% of rows
- **Recoded runtime zeros as NaN**: 30.5% of rows
- **Recoded negative runtime/revenue as NaN**: 0.0001% / 0.0001% of rows
- **Dropped: missing runtime rows**: 384,365
- **Dropped: duplicate id rows**: 418 (kept most complete row per id)
- **Dropped: duplicate imdb_id rows**: 2,250 (kept most complete row per imdb_id, tie-broken by lowest id)
- **Dropped: missing title rows**: 6
- **Dropped: inconsistent vote_average/vote_count rows**: 1,021
- **Final missingness (top 3 by %)**: {'revenue': 97.61, 'budget': 92.92, 'tagline': 80.88}
- **vote_count >= 1 / >= 5 / >= 10**: 291,889 / 121,536 / 77,473 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-12 08:37)

- **Base cleaned dataset (from 01)**: (871113, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7535, 19), 'tmdb_budget_revenue_gt0': (10706, 20), 'tmdb_budget_revenue_gt5': (9174, 19), 'tmdb_budget_revenue_gt10': (8863, 19), 'tmdb_no_budget_revenue_gt0': (291889, 17), 'tmdb_no_budget_revenue_gt5': (107830, 18), 'tmdb_no_budget_revenue_gt10': (72869, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 156, 'tmdb_no_budget_revenue_gt5': 48}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10550, 20), 'tmdb_no_budget_revenue_gt5': (107782, 18)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.3% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.8% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=429.91, p=5.632e-87', 'tmdb_no_budget_revenue_gt5': 'H=1192.53, p=4.978e-251'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the presence/equality flags used then dropped (see Sections 2, 3, 6, 7)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet


## 02 Preprocessing: TMDB candidates -- summary (2026-08-12 09:28)

- **Base cleaned dataset (from 01)**: (871113, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7535, 19), 'tmdb_budget_revenue_gt0': (10706, 20), 'tmdb_budget_revenue_gt5': (9174, 19), 'tmdb_budget_revenue_gt10': (8863, 19), 'tmdb_no_budget_revenue_gt0': (291889, 17), 'tmdb_no_budget_revenue_gt5': (107830, 18), 'tmdb_no_budget_revenue_gt10': (72869, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 156, 'tmdb_no_budget_revenue_gt5': 48}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10550, 20), 'tmdb_no_budget_revenue_gt5': (107782, 18)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.3% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.8% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=429.91, p=5.632e-87', 'tmdb_no_budget_revenue_gt5': 'H=1192.53, p=4.978e-251'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the presence/equality flags used then dropped (see Sections 2, 3, 6, 7)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

