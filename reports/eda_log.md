# EDA Log

## 01 Characterization: TMDB vs Airbnb — combined summary (2026-07-15 00:08)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **Airbnb rows (raw -> cleaned)**: 92638 -> 92638
- **TMDB memory footprint (deep)**: 734.2 MB
- **Airbnb memory footprint (deep)**: 256.4 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'runtime': 0.221, 'budget': 0.179, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 50, ('Airbnb', 'neighbourhood_cleansed'): 17, ('Airbnb', 'room_type'): 3, ('Airbnb', 'property_type'): 46, ('Airbnb', 'host_is_superhost'): 2, ('Airbnb', 'instant_bookable'): 1}
- **TMDB overview: % > 512 tokens (sampled)**: 0.0%
- **Airbnb description: % > 512 tokens (sampled)**: 0.0%
- **TMDB overview: % flagged non-English (sampled)**: 5.0%
- **Airbnb description: % flagged non-English (sampled)**: 0.0%
- **TMDB overview: % HTML-contaminated**: 0.01%
- **Airbnb description: % HTML-contaminated**: 51.56%
- **TMDB genres vocabulary size**: 19
- **TMDB keywords vocabulary size**: 68696
- **Airbnb amenities vocabulary size**: 9606
- **Multi-hot vs pooled-embedding recommendation**: genres: multi-hot feasible; keywords/amenities: use pooled SBERT-of-item-names if vocab_size > 5000 (see Section 5 warnings above)

Figures:
- `reports/figures/01_tmdb_vote_average_hist.png`
- `reports/figures/01_airbnb_price_hist.png`
- `reports/figures/02_tmdb_boxplots.png`
- `reports/figures/02_airbnb_boxplots.png`
- `reports/figures/02_tmdb_correlation_heatmap.png`
- `reports/figures/02_airbnb_correlation_heatmap.png`
- `reports/figures/04_tmdb_overview_length_hist.png`
- `reports/figures/04_airbnb_description_length_hist.png`

---

## 01 Characterization: TMDB vs Airbnb — combined summary (2026-07-15 00:40)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **Airbnb rows (raw -> cleaned)**: 92638 -> 92638
- **TMDB memory footprint (deep)**: 734.2 MB
- **Airbnb memory footprint (deep)**: 256.4 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 50, ('Airbnb', 'neighbourhood_cleansed'): 17, ('Airbnb', 'room_type'): 3, ('Airbnb', 'property_type'): 46, ('Airbnb', 'host_is_superhost'): 2, ('Airbnb', 'instant_bookable'): 1}
- **TMDB overview: % > 512 tokens (sampled)**: 0.0%
- **Airbnb description: % > 512 tokens (sampled)**: 0.0%
- **TMDB overview: % flagged non-English (sampled)**: 5.0%
- **Airbnb description: % flagged non-English (sampled)**: 0.0%
- **TMDB overview: % HTML-contaminated**: 0.01%
- **Airbnb description: % HTML-contaminated**: 51.56%
- **TMDB genres vocabulary size**: 19
- **TMDB keywords vocabulary size**: 68696
- **Airbnb amenities vocabulary size**: 9606
- **Multi-hot vs pooled-embedding recommendation**: genres: multi-hot feasible; keywords/amenities: use pooled SBERT-of-item-names if vocab_size > 5000 (see Section 5 warnings above)

Figures:
- `reports/figures/01_tmdb_vote_average_hist.png`
- `reports/figures/01_airbnb_price_hist.png`
- `reports/figures/02_tmdb_boxplots.png`
- `reports/figures/02_airbnb_boxplots.png`
- `reports/figures/02_tmdb_correlation_heatmap.png`
- `reports/figures/02_airbnb_correlation_heatmap.png`
- `reports/figures/04_tmdb_overview_length_hist.png`
- `reports/figures/04_airbnb_description_length_hist.png`

---

## 01 Characterization: TMDB vs Airbnb — combined summary (2026-07-15 19:20)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **Airbnb rows (raw -> cleaned)**: 92638 -> 92638
- **TMDB memory footprint (deep)**: 734.2 MB
- **Airbnb memory footprint (deep)**: 256.4 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 50, ('Airbnb', 'neighbourhood_cleansed'): 17, ('Airbnb', 'room_type'): 3, ('Airbnb', 'property_type'): 46, ('Airbnb', 'host_is_superhost'): 2, ('Airbnb', 'instant_bookable'): 1}
- **TMDB overview: % > 512 tokens (sampled)**: 0.0%
- **Airbnb description: % > 512 tokens (sampled)**: 0.0%
- **TMDB overview: % flagged non-English (sampled)**: 5.0%
- **Airbnb description: % flagged non-English (sampled)**: 0.0%
- **TMDB overview: % HTML-contaminated**: 0.01%
- **Airbnb description: % HTML-contaminated**: 51.56%
- **TMDB genres vocabulary size**: 19
- **TMDB keywords vocabulary size**: 68696
- **Airbnb amenities vocabulary size**: 9606
- **Multi-hot vs pooled-embedding recommendation**: genres: multi-hot feasible; keywords/amenities: use pooled SBERT-of-item-names if vocab_size > 5000 (see Section 5 warnings above)

Figures:
- `reports/figures/01_tmdb_vote_average_hist.png`
- `reports/figures/01_airbnb_price_hist.png`
- `reports/figures/02_tmdb_boxplots.png`
- `reports/figures/02_airbnb_boxplots.png`
- `reports/figures/02_tmdb_correlation_heatmap.png`
- `reports/figures/02_airbnb_correlation_heatmap.png`
- `reports/figures/04_tmdb_overview_length_hist.png`
- `reports/figures/04_airbnb_description_length_hist.png`

---

## 01 Characterization: TMDB vs Airbnb — combined summary (2026-07-16 20:34)

- **TMDB rows (raw -> released, zero-coded)**: 1456829 -> 1401498
- **Airbnb rows (raw -> cleaned)**: 92638 -> 92638
- **TMDB memory footprint (deep)**: 734.2 MB
- **Airbnb memory footprint (deep)**: 256.4 MB
- **TMDB most predictive numeric features (|r| with vote_average)**: {'budget': 0.179, 'runtime': 0.169, 'popularity': 0.128}
- **Entity-embedding dimension table (see notebook Section 3)**: {('TMDB', 'original_language'): 50, ('Airbnb', 'neighbourhood_cleansed'): 17, ('Airbnb', 'room_type'): 3, ('Airbnb', 'property_type'): 46, ('Airbnb', 'host_is_superhost'): 2, ('Airbnb', 'instant_bookable'): 1}
- **TMDB overview: % > 512 tokens (sampled)**: 0.0%
- **Airbnb description: % > 512 tokens (sampled)**: 0.0%
- **TMDB overview: % flagged non-English (sampled)**: 5.0%
- **Airbnb description: % flagged non-English (sampled)**: 0.0%
- **TMDB overview: % HTML-contaminated**: 0.01%
- **Airbnb description: % HTML-contaminated**: 51.56%
- **TMDB genres vocabulary size**: 19
- **TMDB keywords vocabulary size**: 68696
- **Airbnb amenities vocabulary size**: 9606
- **Multi-hot vs pooled-embedding recommendation**: genres: multi-hot feasible; keywords/amenities: use pooled SBERT-of-item-names if vocab_size > 5000 (see Section 5 warnings above)

Figures:
- `reports/figures/01_tmdb_vote_average_hist.png`
- `reports/figures/01_airbnb_price_hist.png`
- `reports/figures/02_tmdb_boxplots.png`
- `reports/figures/02_airbnb_boxplots.png`
- `reports/figures/02_tmdb_correlation_heatmap.png`
- `reports/figures/02_airbnb_correlation_heatmap.png`
- `reports/figures/04_tmdb_overview_length_hist.png`
- `reports/figures/04_airbnb_description_length_hist.png`

---

