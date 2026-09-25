# Feature Encoding Log

## 04 Feature Encoding: TMDB -- summary (2026-08-19 20:41)

- **Active candidate**: br_gt0
- **Target/feature split**: target=vote_average, id_cols=['id', 'title'], 22 feature_cols
- **Column groups**: {'numeric': ['revenue', 'runtime', 'budget', 'popularity', 'release_year'], 'bypass': ['has_genres', 'has_keywords', 'has_production_companies', 'has_production_countries', 'has_spoken_languages', 'has_overview', 'release_month_sin', 'release_month_cos', 'title_differs_from_original'], 'categorical': ['original_language'], 'text': ['original_title', 'overview'], 'list': ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']}
- **Standardizer**: fit on train, round-trip verified for ['revenue', 'runtime', 'budget', 'popularity', 'release_year']
- **PiecewiseLinearEncoder**: n_bins=32 per numeric column, round-trip verified, hand-computed value matched
- **CategoricalLookup (original_language)**: vocab_size=6 (matches 03_feature_split.ipynb's reported cardinality), round-trip verified
- **Text encoding (SBERT)**: ('overview', 'original_title') cached to C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\encoders\br_gt0 for train/val/test
- **List-valued field pooling (vocab_size per field)**: {'genres': 19, 'production_companies': 2001, 'production_countries': 123, 'spoken_languages': 116, 'keywords': 2001}
- **Raw baseline shape (train)**: (8403, 20)
- **Classical baseline shape (train)**: (8403, 4286)
- **Saved raw/classical baselines**: train/val/test parquet files under C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\models\br_gt0, id column first
- **FittedEncoders bundle dims**: {'numeric': 168, 'original_language': 6, 'overview': 768, 'original_title': 768, 'genres': 19, 'production_companies': 2001, 'production_countries': 123, 'spoken_languages': 116, 'keywords': 2001}
- **FittedEncoders checks**: include_text=False omits text fields; SBERT lookup is id-based, not positional; save()/fit() round-trip identical; transform() cost stable under corruption (max/min ratio=1.03x)
- **Encoders saved to**: C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\encoders\br_gt0

---

## 04 Feature Encoding: TMDB -- summary (2026-08-19 21:01)

- **Active candidate**: br_gt0
- **Target/feature split**: target=vote_average, id_cols=['id', 'title'], 22 feature_cols
- **Column groups**: {'numeric': ['revenue', 'runtime', 'budget', 'popularity', 'release_year'], 'bypass': ['has_genres', 'has_keywords', 'has_production_companies', 'has_production_countries', 'has_spoken_languages', 'has_overview', 'release_month_sin', 'release_month_cos', 'title_differs_from_original'], 'categorical': ['original_language'], 'text': ['original_title', 'overview'], 'list': ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']}
- **Standardizer**: fit on train, round-trip verified for ['revenue', 'runtime', 'budget', 'popularity', 'release_year']
- **PiecewiseLinearEncoder**: n_bins=32 per numeric column, round-trip verified, hand-computed value matched
- **CategoricalLookup (original_language)**: vocab_size=6 (matches 03_feature_split.ipynb's reported cardinality), round-trip verified
- **Text encoding (SBERT)**: ('overview', 'original_title') cached to C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\encoders\br_gt0 for train/val/test
- **List-valued field pooling (vocab_size per field)**: {'genres': 19, 'production_companies': 2001, 'production_countries': 123, 'spoken_languages': 116, 'keywords': 2001}
- **Raw baseline shape (train)**: (8403, 20)
- **Classical baseline shape (train)**: (8403, 4286)
- **Saved raw/classical baselines**: train/val/test parquet files under C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\models\br_gt0, id column first
- **FittedEncoders bundle dims**: {'numeric': 168, 'original_language': 6, 'overview': 768, 'original_title': 768, 'genres': 19, 'production_companies': 2001, 'production_countries': 123, 'spoken_languages': 116, 'keywords': 2001}
- **FittedEncoders checks**: include_text=False omits text fields; SBERT lookup is id-based, not positional; save()/fit() round-trip identical; transform() cost stable under corruption (max/min ratio=1.02x)
- **Encoders saved to**: C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\encoders\br_gt0

---

## 04 Feature Encoding: TMDB -- summary (2026-08-31 16:48)

- **Active candidate**: br_gt0
- **Target/feature split**: target=vote_average, id_cols=['id', 'title'], 22 feature_cols
- **Column groups**: {'numeric': ['revenue', 'runtime', 'budget', 'popularity', 'release_year'], 'bypass': ['has_genres', 'has_keywords', 'has_production_companies', 'has_production_countries', 'has_spoken_languages', 'has_overview', 'release_month_sin', 'release_month_cos', 'title_differs_from_original'], 'categorical': ['original_language'], 'text': ['original_title', 'overview'], 'list': ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']}
- **Standardizer**: fit on train, round-trip verified for ['revenue', 'runtime', 'budget', 'popularity', 'release_year']
- **PiecewiseLinearEncoder**: n_bins=32 per numeric column, round-trip verified, hand-computed value matched
- **CategoricalLookup (original_language)**: vocab_size=6 (matches 03_feature_split.ipynb's reported cardinality), round-trip verified
- **Text encoding (SBERT)**: ('overview', 'original_title') cached to C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\encoders\br_gt0 for train/val/test
- **List-valued field pooling (vocab_size per field)**: {'genres': 19, 'production_companies': 2001, 'production_countries': 123, 'spoken_languages': 116, 'keywords': 2001}
- **Raw baseline shape (train)**: (8403, 20)
- **Classical baseline shape (train)**: (8403, 4286)
- **Saved raw/classical baselines**: train/val/test parquet files under C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\models\br_gt0, id column first
- **FittedEncoders bundle dims**: {'numeric': 168, 'original_language': 6, 'overview': 768, 'original_title': 768, 'genres': 19, 'production_companies': 2001, 'production_countries': 123, 'spoken_languages': 116, 'keywords': 2001}
- **FittedEncoders checks**: include_text=False omits text fields; SBERT lookup is id-based, not positional; save()/fit() round-trip identical; transform() cost stable under corruption (max/min ratio=1.02x)
- **Encoders saved to**: C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\data\encoders\br_gt0

---

