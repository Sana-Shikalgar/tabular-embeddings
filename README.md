# smart-tabular-embeddings
Creating Smart Embeddings for Complex Multi-Type Tabular Datasets


## Environment Setup

This project uses a conda environment (Python 3.11).

```bash
conda env create -f environment.yml
conda activate smart-tabular-embeddings
```

To use this environment as a Jupyter kernel in VS Code / Jupyter, run once
from within the activated environment:

```bash
python -m ipykernel install --user --name smart-tabular-embeddings --display-name "Python (smart-tabular-embeddings)"
```

then select "Python (smart-tabular-embeddings)" as the kernel for notebooks
in `notebooks/`.

## Data Sources & Licensing

This project uses two candidate datasets during exploratory analysis:

- **Full TMDB Movies Dataset 2024 (1M Movies)** — sourced via Kaggle.
  Licensed under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/index.html).
- **Inside Airbnb: London** — sourced from [Inside Airbnb](http://insideairbnb.com/).
  Licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

Raw data files are not included in this repository. See `src/eda/loaders.py`
and the notes below for how to obtain each dataset locally.

## Development Notes

### Notebook outputs and `nbstripout`

This repo has [`nbstripout`](https://github.com/kynan/nbstripout) installed
as a git filter (`nbstripout --install`), which strips notebook cell
outputs from `.ipynb` files at commit time. This means:

- Rendered outputs stay in your local working copy — nothing is deleted on
  disk, and re-running a notebook restores them.
- `git diff` / commits only show code and markdown changes, not embedded
  image/output blobs, so notebook diffs stay readable.
- If a GitHub view of a notebook looks like it's missing output cells,
  that's expected — it's the last *committed* version, which never has
  outputs baked in.

If you want a specific notebook's outputs to actually go into a commit
(e.g. so it renders with output on GitHub), you have two options:

- Temporarily disable the filter for that commit: `git commit --no-verify`
  won't help here since this is a content filter, not a hook — instead run
  `nbstripout --uninstall` first, commit, then optionally re-run
  `nbstripout --install` afterwards to resume stripping.
- Or force-add the notebook bypassing the filter: `git add --renormalize`
  won't restore stripped output either, since stripping happens on the
  `git add`/commit path itself — uninstalling before committing is the
  reliable option.