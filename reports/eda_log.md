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

## 03 Feature Split: TMDB train/val/test -- summary (2026-08-12 09:37)

- **Split sizes (80/10/10)**: {'train': (8440, 20), 'val': (1055, 20), 'test': (1055, 20)}
- **Split sizes (70/15/15)**: {'train': (75447, 18), 'val': (16167, 18), 'test': (16168, 18)}
- **Stratified on**: original_language, via a disposable stratify_key that pools languages under 10 total occurrences into 'other' (see Section 1)
- **Rare-language bucketing (train-derived, min_count=100)**: {'tmdb_budget_revenue_gt0': "5 kept languages + 'other'", 'tmdb_no_budget_revenue_gt5': "35 kept languages + 'other'"}
- **original_language embedding dim (raw 179 -> post-binning)**: {'raw (00_tmdb_eda.ipynb)': 90, 'tmdb_budget_revenue_gt0': 4, 'tmdb_no_budget_revenue_gt5': 19}
- **Vocabulary cap (train-derived top 2000 + 'Other')**: {'tmdb_budget_revenue_gt0 / keywords': 'vocab_size=2001, train coverage=66.6%', 'tmdb_budget_revenue_gt0 / production_companies': 'vocab_size=2001, train coverage=69.2%', 'tmdb_no_budget_revenue_gt5 / keywords': 'vocab_size=2001, train coverage=66.0%', 'tmdb_no_budget_revenue_gt5 / production_companies': 'vocab_size=2001, train coverage=47.8%'}
- **Missing values remaining per split (total cells)**: {'tmdb_br_gt0_train': 0, 'tmdb_br_gt0_val': 0, 'tmdb_br_gt0_test': 0, 'tmdb_nbr_gt5_train': 0, 'tmdb_nbr_gt5_val': 0, 'tmdb_nbr_gt5_test': 0}
- **Saved to**: {'tmdb_br_gt0_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_train.parquet', 'tmdb_br_gt0_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_val.parquet', 'tmdb_br_gt0_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_test.parquet', 'tmdb_nbr_gt5_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_train.parquet', 'tmdb_nbr_gt5_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_val.parquet', 'tmdb_nbr_gt5_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_test.parquet'}

Figures:
- `reports/figures/03_tmdb_budget_revenue_gt0_language_coverage.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_language_coverage.png`
- `reports/figures/03_tmdb_budget_revenue_gt0_vocab_coverage_elbows.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_vocab_coverage_elbows.png`

---

## 00 Characterization: TMDB — summary (2026-08-12 10:10)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 50}
- **TMDB overview: % > 128 tokens (sampled)**: 0.0%
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

## 00 Characterization: TMDB — summary (2026-08-12 10:23)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.4%
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

## 00 Characterization: TMDB — summary (2026-08-12 10:33)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: % flagged non-English (sampled)**: 6.0%
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

## 00 Characterization: TMDB — summary (2026-08-12 11:00)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.4% / 4.2% / 95.4%
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

## 00 Characterization: TMDB — summary (2026-08-12 11:16)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.5% / 4.2% / 95.3%
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

## 00 Characterization: TMDB — summary (2026-08-12 11:26)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.3% / 6.4% / 93.3%
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

## 00 Characterization: TMDB — summary (2026-08-12 11:36)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.3% / 6.4% / 93.3%
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

## 00 Characterization: TMDB — summary (2026-08-13 11:48)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.3% / 6.4% / 93.3%
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

## 00 Characterization: TMDB — summary (2026-08-13 13:05)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.3% / 6.4% / 93.3%
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

## 02 Preprocessing: TMDB candidates -- summary (2026-08-13 21:33)

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

## 01 Overview & Cleaning: TMDB -- summary (2026-08-13 23:23)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 863,858
- **TMDB columns (raw -> cleaned)**: 24 -> 19
- **Raw memory footprint (deep)**: 734.2 MB
- **Dropped: status != 'Released'**: 54,181 rows
- **Dropped: adult == True**: 535,195 rows
- **Recoded budget/revenue zeros as NaN**: 93.8% / 98.1% of rows
- **Recoded runtime zeros as NaN**: 30.5% of rows
- **Recoded negative runtime/revenue as NaN**: 0.0001% / 0.0001% of rows
- **Dropped: missing runtime rows**: 0
- **Dropped: duplicate id rows**: 414 (kept most complete row per id)
- **Dropped: duplicate imdb_id rows**: 2,158 (kept most complete row per imdb_id, tie-broken by lowest id)
- **Dropped: missing title rows**: 6
- **Dropped: inconsistent vote_average/vote_count rows**: 1,017
- **Final missingness (top 3 by %)**: {'revenue': 97.61, 'budget': 92.9, 'tagline': 80.93}
- **vote_count >= 1 / >= 5 / >= 10**: 289,808 / 120,883 / 77,101 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

## 01 Overview & Cleaning: TMDB -- summary (2026-08-13 23:30)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 863,858
- **TMDB columns (raw -> cleaned)**: 24 -> 19
- **Raw memory footprint (deep)**: 734.2 MB
- **Dropped: status != 'Released'**: 54,181 rows
- **Dropped: adult == True**: 535,195 rows
- **Recoded budget/revenue zeros as NaN**: 93.8% / 98.1% of rows
- **Recoded runtime zeros as NaN**: 30.5% of rows
- **Recoded negative runtime/revenue as NaN**: 0.0001% / 0.0001% of rows
- **Dropped: runtime > 200 minutes rows**: 31.11% of rows dropped (runtime > 200)
- **Dropped: missing runtime rows**: 0
- **Dropped: duplicate id rows**: 414 (kept most complete row per id)
- **Dropped: duplicate imdb_id rows**: 2,158 (kept most complete row per imdb_id, tie-broken by lowest id)
- **Dropped: missing title rows**: 6
- **Dropped: inconsistent vote_average/vote_count rows**: 1,017
- **Final missingness (top 3 by %)**: {'revenue': 97.61, 'budget': 92.9, 'tagline': 80.93}
- **vote_count >= 1 / >= 5 / >= 10**: 289,808 / 120,883 / 77,101 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-13 23:36)

- **Base cleaned dataset (from 01)**: (863858, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7510, 19), 'tmdb_budget_revenue_gt0': (10657, 20), 'tmdb_budget_revenue_gt5': (9148, 19), 'tmdb_budget_revenue_gt10': (8838, 19), 'tmdb_no_budget_revenue_gt0': (289808, 17), 'tmdb_no_budget_revenue_gt5': (107266, 18), 'tmdb_no_budget_revenue_gt10': (72524, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 153, 'tmdb_no_budget_revenue_gt5': 47}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10504, 20), 'tmdb_no_budget_revenue_gt5': (107219, 18)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.2% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.9% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=435.69, p=3.282e-88', 'tmdb_no_budget_revenue_gt5': 'H=1190.10, p=1.667e-250'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the presence/equality flags used then dropped (see Sections 2, 3, 6, 7)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-13 23:44)

- **Base cleaned dataset (from 01)**: (863858, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7510, 19), 'tmdb_budget_revenue_gt0': (10657, 20), 'tmdb_budget_revenue_gt5': (9148, 19), 'tmdb_budget_revenue_gt10': (8838, 19), 'tmdb_no_budget_revenue_gt0': (289808, 17), 'tmdb_no_budget_revenue_gt5': (107266, 18), 'tmdb_no_budget_revenue_gt10': (72524, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 153, 'tmdb_no_budget_revenue_gt5': 47}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10504, 20), 'tmdb_no_budget_revenue_gt5': (107219, 18)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.2% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.9% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=435.69, p=3.282e-88', 'tmdb_no_budget_revenue_gt5': 'H=1190.10, p=1.667e-250'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the presence/equality flags used then dropped (see Sections 2, 3, 6, 7)
- **Log-transformed for skew**: popularity (log1p, both candidates); revenue/budget (log, tmdb_budget_revenue_gt0 only) -- see Section 9 before/after skewness tables
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

## 01 Overview & Cleaning: TMDB -- summary (2026-08-14 00:05)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 863,858
- **TMDB columns (raw -> cleaned)**: 24 -> 19
- **Raw memory footprint (deep)**: 734.2 MB
- **Dropped: status != 'Released'**: 54,181 rows
- **Dropped: adult == True**: 535,195 rows
- **Recoded budget/revenue zeros as NaN**: 93.8% / 98.1% of rows
- **Recoded runtime zeros as NaN**: 30.5% of rows
- **Recoded negative runtime/revenue as NaN**: 0.0001% / 0.0001% of rows
- **Dropped: runtime > 200 minutes rows**: 31.11% of rows dropped (runtime > 200)
- **Dropped: missing runtime rows**: 0
- **Dropped: duplicate id rows**: 414 (kept most complete row per id)
- **Dropped: duplicate imdb_id rows**: 2,158 (kept most complete row per imdb_id, tie-broken by lowest id)
- **Dropped: missing title rows**: 6
- **Dropped: inconsistent vote_average/vote_count rows**: 1,017
- **Final missingness (top 3 by %)**: {'revenue': 97.61, 'budget': 92.9, 'tagline': 80.93}
- **vote_count >= 1 / >= 5 / >= 10**: 289,808 / 120,883 / 77,101 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-14 00:18)

- **Base cleaned dataset (from 01)**: (863858, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7510, 19), 'tmdb_budget_revenue_gt0': (10657, 20), 'tmdb_budget_revenue_gt5': (9148, 19), 'tmdb_budget_revenue_gt10': (8838, 19), 'tmdb_no_budget_revenue_gt0': (289808, 17), 'tmdb_no_budget_revenue_gt5': (107266, 18), 'tmdb_no_budget_revenue_gt10': (72524, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 153, 'tmdb_no_budget_revenue_gt5': 47}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10504, 20), 'tmdb_no_budget_revenue_gt5': (107219, 18)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.2% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.9% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=435.69, p=3.282e-88', 'tmdb_no_budget_revenue_gt5': 'H=1190.10, p=1.667e-250'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the presence/equality flags used then dropped (see Sections 2, 3, 6, 7)
- **Log-transformed for skew**: popularity (log1p, both candidates); revenue/budget (log, tmdb_budget_revenue_gt0 only) -- see Section 9 before/after skewness tables
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

## 03 Feature Split: TMDB train/val/test -- summary (2026-08-14 00:22)

- **Split sizes (80/10/10)**: {'train': (8403, 20), 'val': (1050, 20), 'test': (1051, 20)}
- **Split sizes (70/15/15)**: {'train': (75053, 18), 'val': (16083, 18), 'test': (16083, 18)}
- **Stratified on**: original_language, via a disposable stratify_key that pools languages under 10 total occurrences into 'other' (see Section 1)
- **Rare-language bucketing (train-derived, min_count=100)**: {'tmdb_budget_revenue_gt0': "5 kept languages + 'other'", 'tmdb_no_budget_revenue_gt5': "35 kept languages + 'other'"}
- **original_language embedding dim (raw 179 -> post-binning)**: {'raw (00_tmdb_eda.ipynb)': 90, 'tmdb_budget_revenue_gt0': 4, 'tmdb_no_budget_revenue_gt5': 19}
- **Vocabulary cap (train-derived top 2000 + 'Other')**: {'tmdb_budget_revenue_gt0 / keywords': 'vocab_size=2001, train coverage=66.8%', 'tmdb_budget_revenue_gt0 / production_companies': 'vocab_size=2001, train coverage=69.1%', 'tmdb_no_budget_revenue_gt5 / keywords': 'vocab_size=2001, train coverage=66.0%', 'tmdb_no_budget_revenue_gt5 / production_companies': 'vocab_size=2001, train coverage=47.8%'}
- **Missing values remaining per split (total cells)**: {'tmdb_br_gt0_train': 0, 'tmdb_br_gt0_val': 0, 'tmdb_br_gt0_test': 0, 'tmdb_nbr_gt5_train': 0, 'tmdb_nbr_gt5_val': 0, 'tmdb_nbr_gt5_test': 0}
- **Saved to**: {'tmdb_br_gt0_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_train.parquet', 'tmdb_br_gt0_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_val.parquet', 'tmdb_br_gt0_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_test.parquet', 'tmdb_nbr_gt5_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_train.parquet', 'tmdb_nbr_gt5_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_val.parquet', 'tmdb_nbr_gt5_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_test.parquet'}

Figures:
- `reports/figures/03_tmdb_budget_revenue_gt0_language_coverage.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_language_coverage.png`
- `reports/figures/03_tmdb_budget_revenue_gt0_vocab_coverage_elbows.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_vocab_coverage_elbows.png`

---

## 01 Overview & Cleaning: TMDB -- summary (2026-08-14 00:25)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 863,858
- **TMDB columns (raw -> cleaned)**: 24 -> 19
- **Raw memory footprint (deep)**: 734.2 MB
- **Dropped: status != 'Released'**: 54,181 rows
- **Dropped: adult == True**: 535,195 rows
- **Recoded budget/revenue zeros as NaN**: 93.8% / 98.1% of rows
- **Recoded runtime zeros as NaN**: 30.5% of rows
- **Recoded negative runtime/revenue as NaN**: 0.0001% / 0.0001% of rows
- **Dropped: runtime > 200 minutes rows**: 31.11% of rows dropped (runtime > 200)
- **Dropped: missing runtime rows**: 0
- **Dropped: duplicate id rows**: 414 (kept most complete row per id)
- **Dropped: duplicate imdb_id rows**: 2,158 (kept most complete row per imdb_id, tie-broken by lowest id)
- **Dropped: missing title rows**: 6
- **Dropped: inconsistent vote_average/vote_count rows**: 1,017
- **Final missingness (top 3 by %)**: {'revenue': 97.61, 'budget': 92.9, 'tagline': 80.93}
- **vote_count >= 1 / >= 5 / >= 10**: 289,808 / 120,883 / 77,101 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-14 00:25)

- **Base cleaned dataset (from 01)**: (863858, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7510, 19), 'tmdb_budget_revenue_gt0': (10657, 20), 'tmdb_budget_revenue_gt5': (9148, 19), 'tmdb_budget_revenue_gt10': (8838, 19), 'tmdb_no_budget_revenue_gt0': (289808, 17), 'tmdb_no_budget_revenue_gt5': (107266, 18), 'tmdb_no_budget_revenue_gt10': (72524, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 153, 'tmdb_no_budget_revenue_gt5': 47}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10504, 20), 'tmdb_no_budget_revenue_gt5': (107219, 18)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.2% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.9% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=435.69, p=3.282e-88', 'tmdb_no_budget_revenue_gt5': 'H=1190.10, p=1.667e-250'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the presence/equality flags used then dropped (see Sections 2, 3, 6, 7)
- **Log-transformed for skew**: popularity (log1p, both candidates); revenue/budget (log, tmdb_budget_revenue_gt0 only) -- see Section 9 before/after skewness tables
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

## 03 Feature Split: TMDB train/val/test -- summary (2026-08-14 00:26)

- **Split sizes (80/10/10)**: {'train': (8403, 20), 'val': (1050, 20), 'test': (1051, 20)}
- **Split sizes (70/15/15)**: {'train': (75053, 18), 'val': (16083, 18), 'test': (16083, 18)}
- **Stratified on**: original_language, via a disposable stratify_key that pools languages under 10 total occurrences into 'other' (see Section 1)
- **Rare-language bucketing (train-derived, min_count=100)**: {'tmdb_budget_revenue_gt0': "5 kept languages + 'other'", 'tmdb_no_budget_revenue_gt5': "35 kept languages + 'other'"}
- **original_language embedding dim (raw 179 -> post-binning)**: {'raw (00_tmdb_eda.ipynb)': 90, 'tmdb_budget_revenue_gt0': 4, 'tmdb_no_budget_revenue_gt5': 19}
- **Vocabulary cap (train-derived top 2000 + 'Other')**: {'tmdb_budget_revenue_gt0 / keywords': 'vocab_size=2001, train coverage=66.8%', 'tmdb_budget_revenue_gt0 / production_companies': 'vocab_size=2001, train coverage=69.1%', 'tmdb_no_budget_revenue_gt5 / keywords': 'vocab_size=2001, train coverage=66.0%', 'tmdb_no_budget_revenue_gt5 / production_companies': 'vocab_size=2001, train coverage=47.8%'}
- **Missing values remaining per split (total cells)**: {'tmdb_br_gt0_train': 0, 'tmdb_br_gt0_val': 0, 'tmdb_br_gt0_test': 0, 'tmdb_nbr_gt5_train': 0, 'tmdb_nbr_gt5_val': 0, 'tmdb_nbr_gt5_test': 0}
- **Saved to**: {'tmdb_br_gt0_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_train.parquet', 'tmdb_br_gt0_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_val.parquet', 'tmdb_br_gt0_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_test.parquet', 'tmdb_nbr_gt5_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_train.parquet', 'tmdb_nbr_gt5_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_val.parquet', 'tmdb_nbr_gt5_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_test.parquet'}

Figures:
- `reports/figures/03_tmdb_budget_revenue_gt0_language_coverage.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_language_coverage.png`
- `reports/figures/03_tmdb_budget_revenue_gt0_vocab_coverage_elbows.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_vocab_coverage_elbows.png`

---

## 03 Feature Split: TMDB train/val/test -- summary (2026-08-14 08:20)

- **Split sizes (80/10/10)**: {'train': (8403, 20), 'val': (1050, 20), 'test': (1051, 20)}
- **Split sizes (70/15/15)**: {'train': (75053, 18), 'val': (16083, 18), 'test': (16083, 18)}
- **Stratified on**: original_language, via a disposable stratify_key that pools languages under 10 total occurrences into 'other' (see Section 1)
- **Rare-language bucketing (train-derived, min_count=100)**: {'tmdb_budget_revenue_gt0': "5 kept languages + 'other'", 'tmdb_no_budget_revenue_gt5': "35 kept languages + 'other'"}
- **original_language embedding dim (raw 179 -> post-binning)**: {'raw (00_tmdb_eda.ipynb)': 90, 'tmdb_budget_revenue_gt0': 4, 'tmdb_no_budget_revenue_gt5': 19}
- **Vocabulary cap (train-derived top 2000 + 'Other')**: {'tmdb_budget_revenue_gt0 / keywords': 'vocab_size=2001, train coverage=66.8%', 'tmdb_budget_revenue_gt0 / production_companies': 'vocab_size=2001, train coverage=69.1%', 'tmdb_no_budget_revenue_gt5 / keywords': 'vocab_size=2001, train coverage=66.0%', 'tmdb_no_budget_revenue_gt5 / production_companies': 'vocab_size=2001, train coverage=47.8%'}
- **Missing values remaining per split (total cells)**: {'tmdb_br_gt0_train': 0, 'tmdb_br_gt0_val': 0, 'tmdb_br_gt0_test': 0, 'tmdb_nbr_gt5_train': 0, 'tmdb_nbr_gt5_val': 0, 'tmdb_nbr_gt5_test': 0}
- **Saved to**: {'tmdb_br_gt0_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_train.parquet', 'tmdb_br_gt0_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_val.parquet', 'tmdb_br_gt0_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_test.parquet', 'tmdb_nbr_gt5_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_train.parquet', 'tmdb_nbr_gt5_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_val.parquet', 'tmdb_nbr_gt5_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_test.parquet'}

Figures:
- `reports/figures/03_tmdb_budget_revenue_gt0_language_coverage.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_language_coverage.png`
- `reports/figures/03_tmdb_budget_revenue_gt0_vocab_coverage_elbows.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_vocab_coverage_elbows.png`

---

## 03 Feature Split: TMDB train/val/test -- summary (2026-08-14 08:21)

- **Split sizes (80/10/10)**: {'train': (8403, 20), 'val': (1050, 20), 'test': (1051, 20)}
- **Split sizes (70/15/15)**: {'train': (75053, 18), 'val': (16083, 18), 'test': (16083, 18)}
- **Stratified on**: original_language, via a disposable stratify_key that pools languages under 10 total occurrences into 'other' (see Section 1)
- **Rare-language bucketing (train-derived, min_count=100)**: {'tmdb_budget_revenue_gt0': "5 kept languages + 'other'", 'tmdb_no_budget_revenue_gt5': "35 kept languages + 'other'"}
- **original_language embedding dim (raw 179 -> post-binning)**: {'raw (00_tmdb_eda.ipynb)': 90, 'tmdb_budget_revenue_gt0': 4, 'tmdb_no_budget_revenue_gt5': 19}
- **Vocabulary cap (train-derived top 2000 + 'Other')**: {'tmdb_budget_revenue_gt0 / keywords': 'vocab_size=2001, train coverage=66.8%', 'tmdb_budget_revenue_gt0 / production_companies': 'vocab_size=2001, train coverage=69.1%', 'tmdb_no_budget_revenue_gt5 / keywords': 'vocab_size=2001, train coverage=66.0%', 'tmdb_no_budget_revenue_gt5 / production_companies': 'vocab_size=2001, train coverage=47.8%'}
- **Missing values remaining per split (total cells)**: {'tmdb_br_gt0_train': 0, 'tmdb_br_gt0_val': 0, 'tmdb_br_gt0_test': 0, 'tmdb_nbr_gt5_train': 0, 'tmdb_nbr_gt5_val': 0, 'tmdb_nbr_gt5_test': 0}
- **Saved to**: {'tmdb_br_gt0_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_train.parquet', 'tmdb_br_gt0_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_val.parquet', 'tmdb_br_gt0_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_test.parquet', 'tmdb_nbr_gt5_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_train.parquet', 'tmdb_nbr_gt5_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_val.parquet', 'tmdb_nbr_gt5_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_test.parquet'}

Figures:
- `reports/figures/03_tmdb_budget_revenue_gt0_language_coverage.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_language_coverage.png`
- `reports/figures/03_tmdb_budget_revenue_gt0_vocab_coverage_elbows.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_vocab_coverage_elbows.png`

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-14 09:37)

- **Base cleaned dataset (from 01)**: (863858, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7510, 19), 'tmdb_budget_revenue_gt0': (10657, 20), 'tmdb_budget_revenue_gt5': (9148, 19), 'tmdb_budget_revenue_gt10': (8838, 19), 'tmdb_no_budget_revenue_gt0': (289808, 17), 'tmdb_no_budget_revenue_gt5': (107266, 18), 'tmdb_no_budget_revenue_gt10': (72524, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 153, 'tmdb_no_budget_revenue_gt5': 47}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10504, 21), 'tmdb_no_budget_revenue_gt5': (107219, 19)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.2% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.9% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **title_differs_from_original (retained feature - case-sensitive)**: 18.1% / 35.2%
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=435.69, p=3.282e-88', 'tmdb_no_budget_revenue_gt5': 'H=1190.10, p=1.667e-250'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the tagline/overview presence flags and the cs equality flag (see Sections 2, 3, 6, 7); the ci equality flag is kept, renamed to title_differs_from_original
- **Log-transformed for skew**: popularity (log1p, both candidates); revenue/budget (log, tmdb_budget_revenue_gt0 only) -- see Section 9 before/after skewness tables
- **Saved to**: c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, c:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

## 03 Feature Split: TMDB train/val/test -- summary (2026-08-14 09:38)

- **Split sizes (80/10/10)**: {'train': (8403, 21), 'val': (1050, 21), 'test': (1051, 21)}
- **Split sizes (70/15/15)**: {'train': (75053, 19), 'val': (16083, 19), 'test': (16083, 19)}
- **Stratified on**: original_language, via a disposable stratify_key that pools languages under 10 total occurrences into 'other' (see Section 1)
- **Rare-language bucketing (train-derived, min_count=100)**: {'tmdb_budget_revenue_gt0': "5 kept languages + 'other'", 'tmdb_no_budget_revenue_gt5': "35 kept languages + 'other'"}
- **original_language embedding dim (raw 179 -> post-binning)**: {'raw (00_tmdb_eda.ipynb)': 90, 'tmdb_budget_revenue_gt0': 4, 'tmdb_no_budget_revenue_gt5': 19}
- **Vocabulary cap (train-derived top 2000 + 'Other')**: {'tmdb_budget_revenue_gt0 / keywords': 'vocab_size=2001, train coverage=66.8%', 'tmdb_budget_revenue_gt0 / production_companies': 'vocab_size=2001, train coverage=69.1%', 'tmdb_no_budget_revenue_gt5 / keywords': 'vocab_size=2001, train coverage=66.0%', 'tmdb_no_budget_revenue_gt5 / production_companies': 'vocab_size=2001, train coverage=47.8%'}
- **Missing values remaining per split (total cells)**: {'tmdb_br_gt0_train': 0, 'tmdb_br_gt0_val': 0, 'tmdb_br_gt0_test': 0, 'tmdb_nbr_gt5_train': 0, 'tmdb_nbr_gt5_val': 0, 'tmdb_nbr_gt5_test': 0}
- **Saved to**: {'tmdb_br_gt0_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_train.parquet', 'tmdb_br_gt0_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_val.parquet', 'tmdb_br_gt0_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_test.parquet', 'tmdb_nbr_gt5_train': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_train.parquet', 'tmdb_nbr_gt5_val': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_val.parquet', 'tmdb_nbr_gt5_test': 'c:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_test.parquet'}

Figures:
- `reports/figures/03_tmdb_budget_revenue_gt0_language_coverage.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_language_coverage.png`
- `reports/figures/03_tmdb_budget_revenue_gt0_vocab_coverage_elbows.png`
- `reports/figures/03_tmdb_no_budget_revenue_gt5_vocab_coverage_elbows.png`

---

## 01 Overview & Cleaning: TMDB -- summary (2026-08-14 11:34)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 863,858
- **TMDB columns (raw -> cleaned)**: 24 -> 19
- **Raw memory footprint (deep)**: 734.2 MB
- **Dropped: status != 'Released'**: 54,181 rows
- **Dropped: adult == True**: 535,195 rows
- **Recoded budget/revenue zeros as NaN**: 93.8% / 98.1% of rows
- **Recoded runtime zeros as NaN**: 30.5% of rows
- **Recoded negative runtime/revenue as NaN**: 0.0001% / 0.0001% of rows
- **Dropped: runtime > 200 minutes rows**: 31.11% of rows dropped (runtime > 200)
- **Dropped: missing runtime rows**: 0
- **Dropped: duplicate id rows**: 414 (kept most complete row per id)
- **Dropped: duplicate imdb_id rows**: 2,158 (kept most complete row per imdb_id, tie-broken by lowest id)
- **Dropped: missing title rows**: 6
- **Dropped: inconsistent vote_average/vote_count rows**: 1,017
- **Final missingness (top 3 by %)**: {'revenue': 97.61, 'budget': 92.9, 'tagline': 80.93}
- **vote_count >= 1 / >= 5 / >= 10**: 289,808 / 120,883 / 77,101 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01_tmdb_vote_average_hist.png`

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-14 11:35)

- **Base cleaned dataset (from 01)**: (863858, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7510, 19), 'tmdb_budget_revenue_gt0': (10657, 20), 'tmdb_budget_revenue_gt5': (9148, 19), 'tmdb_budget_revenue_gt10': (8838, 19), 'tmdb_no_budget_revenue_gt0': (289808, 17), 'tmdb_no_budget_revenue_gt5': (107266, 18), 'tmdb_no_budget_revenue_gt10': (72524, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 153, 'tmdb_no_budget_revenue_gt5': 47}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10504, 21), 'tmdb_no_budget_revenue_gt5': (107219, 19)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.2% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.9% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **title_differs_from_original (retained feature - case-sensitive)**: 18.1% / 35.2%
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=435.69, p=3.282e-88', 'tmdb_no_budget_revenue_gt5': 'H=1190.10, p=1.667e-250'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the tagline/overview presence flags and the cs equality flag (see Sections 2, 3, 6, 7); the ci equality flag is kept, renamed to title_differs_from_original
- **Log-transformed for skew**: popularity (log1p, both candidates); revenue/budget (log, tmdb_budget_revenue_gt0 only) -- see Section 9 before/after skewness tables
- **Saved to**: C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

## 00 Characterization: TMDB — summary (2026-08-14 11:36)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.3% / 6.4% / 93.3%
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

## 00 Characterization: TMDB — summary (2026-08-19 16:40)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.3% / 6.4% / 93.3%
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

## 01 Overview & Cleaning: TMDB -- summary (2026-08-19 17:30)

- **TMDB rows (raw -> cleaned)**: 1,456,829 -> 863,858
- **TMDB columns (raw -> cleaned)**: 24 -> 19
- **Raw memory footprint (deep)**: 734.2 MB
- **Dropped: status != 'Released'**: 54,181 rows
- **Dropped: adult == True**: 535,195 rows
- **Recoded budget/revenue zeros as NaN**: 93.8% / 98.1% of rows
- **Recoded runtime zeros as NaN**: 30.5% of rows
- **Recoded negative runtime/revenue as NaN**: 0.0001% / 0.0001% of rows
- **Dropped: runtime > 200 minutes rows**: 31.11% of rows dropped (runtime > 200)
- **Dropped: missing runtime rows**: 0
- **Dropped: duplicate id rows**: 414 (kept most complete row per id)
- **Dropped: duplicate imdb_id rows**: 2,158 (kept most complete row per imdb_id, tie-broken by lowest id)
- **Dropped: missing title rows**: 6
- **Dropped: inconsistent vote_average/vote_count rows**: 1,017
- **Final missingness (top 3 by %)**: {'revenue': 97.61, 'budget': 92.9, 'tagline': 80.93}
- **vote_count >= 1 / >= 5 / >= 10**: 289,808 / 120,883 / 77,101 rows (vote_count filter deferred to modeling stage, not yet applied)
- **Saved to**: C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_clean.parquet

Figures:
- `reports/figures/01/01_tmdb_missingness_matrix_cleandata.png`
- `reports/figures/01/01_tmdb_vote_average_hist.png`

---

## 02 Preprocessing: TMDB candidates -- summary (2026-08-19 17:31)

- **Base cleaned dataset (from 01)**: (863858, 19)
- **7 candidate shapes**: {'tmdb_no_nulls': (7510, 19), 'tmdb_budget_revenue_gt0': (10657, 20), 'tmdb_budget_revenue_gt5': (9148, 19), 'tmdb_budget_revenue_gt10': (8838, 19), 'tmdb_no_budget_revenue_gt0': (289808, 17), 'tmdb_no_budget_revenue_gt5': (107266, 18), 'tmdb_no_budget_revenue_gt10': (72524, 17)}
- **Rows dropped for missing release_date**: {'tmdb_budget_revenue_gt0': 153, 'tmdb_no_budget_revenue_gt5': 47}
- **Chosen candidates, final shape**: {'tmdb_budget_revenue_gt0': (10504, 25), 'tmdb_no_budget_revenue_gt5': (107219, 23)}
- **tagline/overview presence (overall)**: {'tmdb_budget_revenue_gt0': '80.2% / 99.7%', 'tmdb_no_budget_revenue_gt5': '39.9% / 99.0%'}
- **overview missing values**: filled with "" (empty string), not left as NaN (see Section 4)
- **title == original_title (cs/ci)**: {'tmdb_budget_revenue_gt0': '81.7% / 81.9%', 'tmdb_no_budget_revenue_gt5': '64.6% / 64.8%'}
- **title_differs_from_original (retained feature - case-sensitive)**: 18.1% / 35.2%
- **Kruskal-Wallis on release_year by language**: {'tmdb_budget_revenue_gt0': 'H=435.69, p=3.282e-88', 'tmdb_no_budget_revenue_gt5': 'H=1190.10, p=1.667e-250'}
- **release_date decomposition**: release_year + release_month_sin/cos extracted; release_date and raw release_month then dropped (see Section 5)
- **Dropped along the way**: tagline, has_overview, vote_count, imdb_id, plus the tagline/overview presence flags and the cs equality flag (see Sections 2, 3, 6, 7); the ci equality flag is kept, renamed to title_differs_from_original
- **Log-transformed for skew**: popularity (log1p, both candidates); revenue/budget (log, tmdb_budget_revenue_gt0 only) -- see Section 9 before/after skewness tables
- **Saved to**: C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_br_gt0.parquet, C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\processed\tmdb_nbr_gt5.parquet

---

## 03 Feature Split: TMDB train/val/test -- summary (2026-08-19 17:31)

- **Split sizes (80/10/10)**: {'train': (8403, 25), 'val': (1050, 25), 'test': (1051, 25)}
- **Split sizes (70/15/15)**: {'train': (75053, 23), 'val': (16083, 23), 'test': (16083, 23)}
- **Stratified on**: original_language, via a disposable stratify_key that pools languages under 10 total occurrences into 'other' (see Section 1)
- **Rare-language bucketing (train-derived, min_count=100)**: {'tmdb_budget_revenue_gt0': "5 kept languages + 'other'", 'tmdb_no_budget_revenue_gt5': "35 kept languages + 'other'"}
- **original_language embedding dim (raw 179 -> post-binning)**: {'raw (00_tmdb_eda.ipynb)': 90, 'tmdb_budget_revenue_gt0': 4, 'tmdb_no_budget_revenue_gt5': 19}
- **Vocabulary cap (train-derived top 2000 + 'Other')**: {'tmdb_budget_revenue_gt0 / keywords': 'vocab_size=2001, train coverage=66.8%', 'tmdb_budget_revenue_gt0 / production_companies': 'vocab_size=2001, train coverage=69.1%', 'tmdb_no_budget_revenue_gt5 / keywords': 'vocab_size=2001, train coverage=66.0%', 'tmdb_no_budget_revenue_gt5 / production_companies': 'vocab_size=2001, train coverage=47.8%'}
- **Missing values remaining per split (total cells)**: {'tmdb_br_gt0_train': 0, 'tmdb_br_gt0_val': 0, 'tmdb_br_gt0_test': 0, 'tmdb_nbr_gt5_train': 0, 'tmdb_nbr_gt5_val': 0, 'tmdb_nbr_gt5_test': 0}
- **Saved to**: {'tmdb_br_gt0_train': 'C:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_train.parquet', 'tmdb_br_gt0_val': 'C:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_val.parquet', 'tmdb_br_gt0_test': 'C:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_br_gt0_test.parquet', 'tmdb_nbr_gt5_train': 'C:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_train.parquet', 'tmdb_nbr_gt5_val': 'C:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_val.parquet', 'tmdb_nbr_gt5_test': 'C:\\Users\\knowu\\Documents\\Project-Repos\\Dissertation\\smart-tabular-embeddings\\data\\final\\tmdb_nbr_gt5_test.parquet'}

Figures:
- `reports/figures/03/03_tmdb_budget_revenue_gt0_language_coverage.png`
- `reports/figures/03/03_tmdb_no_budget_revenue_gt5_language_coverage.png`
- `reports/figures/03/03_tmdb_budget_revenue_gt0_vocab_coverage_elbows.png`
- `reports/figures/03/03_tmdb_no_budget_revenue_gt5_vocab_coverage_elbows.png`

---

## 00 Characterization: TMDB -- summary (2026-08-19 17:32)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **TMDB memory footprint (deep)**: 734.2 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 90}
- **TMDB overview: % > 128 tokens (sampled)**: 12.2%
- **TMDB overview: non-English / too short / uncertain (sampled)**: 0.2% / 6.4% / 93.4%
- **TMDB overview: % HTML-contaminated**: 0.01%
- **TMDB genres vocabulary size**: 19
- **TMDB keywords vocabulary size**: 68696
- **Multi-hot vs pooled-embedding recommendation**: genres: multi-hot feasible; keywords: use pooled SBERT-of-item-names if vocab_size > 5000 (see Section 5 warnings above)

Figures:
- `reports/figures/00/00_tmdb_missingness_matrix.png`
- `reports/figures/00/00_tmdb_vote_average_hist.png`
- `reports/figures/00/00_tmdb_histograms.png`
- `reports/figures/00/00_tmdb_boxplots.png`
- `reports/figures/00/00_tmdb_correlation_heatmap.png`
- `reports/figures/00/00_tmdb_original_language_long_tail.png`
- `reports/figures/00/00_tmdb_overview_length_hist.png`
- `reports/figures/00/00_tmdb_tagline_length_hist.png`
- `reports/figures/00/00_tmdb_title_length_hist.png`
- `reports/figures/00/00_tmdb_original_title_length_hist.png`

---

