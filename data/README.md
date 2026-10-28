# data/ (local only, git-ignored)

Raw downloads, extracted landmarks and WebDataset shards live here on your machine (or in MinIO). Nothing in this folder is committed.

```
data/
├── raw/<source-slug>/          # exactly as downloaded (kaggle, huggingface-cli, zenodo)
├── landmarks/<source-slug>/    # (T,107,3) float16 .npy per sample + meta.json
└── shards/<build-name>/<version>/{train,val,test}-NNNNNN.tar
```

* Download commands: `tools/scripts/download_sources.sh`.
* Every source must exist in `dataset.sources` with its licence before `tools/scripts/import_source.py` will load it.
