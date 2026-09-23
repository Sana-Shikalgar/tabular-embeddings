# Embedding Paradigms Log

## 05 Embedding Paradigms: TMDB -- summary (2026-08-20 10:10)

- **Active candidate**: br_gt0
- **FT-Transformer**: {'parameters': 1894465, 'epochs run': 58, 'final train MSE': np.float64(0.2068), 'val MSE (restored/best epoch)': np.float64(1.0056)}
- **FT-Transformer (text-free)**: {'parameters': 1599169, 'epochs run': 55, 'final train MSE': np.float64(0.6616), 'val MSE (restored/best epoch)': np.float64(0.9637)}
- **SubTab**: {'subset_width': 567, 'full_width': 2270, 'parameters': 5007582, 'epochs run': 24, 'val L_r (restored/best epoch)': np.float64(0.0166)}
- **SubTab (text-free)**: {'subset_width': 182, 'full_width': 734, 'parameters': 3038942, 'epochs run': 35, 'val L_r (restored/best epoch)': np.float64(0.0289)}
- **SCARF**: {'flat_width': 2671, 'parameters': 1831872, 'epochs run': 53, 'val NT-Xent (restored/best epoch)': np.float64(3.8485)}
- **SCARF (text-free)**: {'flat_width': 1135, 'parameters': 1438656, 'epochs run': 100, 'val NT-Xent (restored/best epoch)': np.float64(4.3039)}
- **Saved embeddings**: train/val/test parquet files under C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\models\br_gt0, id column first, for ft_transformer/subtab/scarf and their _no_text counterparts

Figures:
- `reports/figures/05/05_ft_transformer_training_curve.png`
- `reports/figures/05/05_subtab_svd_cumulative_variance_keywords.png`
- `reports/figures/05/05_subtab_svd_cumulative_variance_production_companies.png`
- `reports/figures/05/05_cross_paradigm_training_curves.png`

---

## 05 Embedding Paradigms: TMDB -- summary (2026-09-23 23:22)

- **Active candidate**: br_gt0
- **FT-Transformer**: {'parameters': 1894465, 'epochs run': 58, 'final train MSE': np.float64(0.2068), 'val MSE (restored/best epoch)': np.float64(1.0056)}
- **FT-Transformer (text-free)**: {'parameters': 1599169, 'epochs run': 55, 'final train MSE': np.float64(0.6616), 'val MSE (restored/best epoch)': np.float64(0.9637)}
- **SubTab**: {'subset_width': 567, 'full_width': 2270, 'parameters': 5007582, 'epochs run': 24, 'val L_r (restored/best epoch)': np.float64(0.0166)}
- **SubTab (text-free)**: {'subset_width': 182, 'full_width': 734, 'parameters': 3038942, 'epochs run': 35, 'val L_r (restored/best epoch)': np.float64(0.0289)}
- **SCARF**: {'flat_width': 2671, 'parameters': 1831872, 'epochs run': 53, 'val NT-Xent (restored/best epoch)': np.float64(3.8485)}
- **SCARF (text-free)**: {'flat_width': 1135, 'parameters': 1438656, 'epochs run': 100, 'val NT-Xent (restored/best epoch)': np.float64(4.3039)}
- **Saved embeddings**: train/val/test parquet files under C:\Users\knowu\Documents\Project-Repos\Dissertation\smart-tabular-embeddings\models\br_gt0, id column first, for ft_transformer/subtab/scarf and their _no_text counterparts

Figures:
- `reports/figures/05/05_ft_transformer_training_curve.png`
- `reports/figures/05/05_subtab_svd_cumulative_variance_keywords.png`
- `reports/figures/05/05_subtab_svd_cumulative_variance_production_companies.png`
- `reports/figures/05/05_cross_paradigm_training_curves.png`

---

