# Streamlit Community Cloud deployment

This repository is locked to `scikit-learn==1.9.0`, which has wheels for Python 3.14.
Do not change it back to 1.7.2.

## Deploy

1. Replace the repository contents with this project.
2. Commit and push all changes, including `uv.lock` and `artifacts/model_bundle.meta.json`.
3. In Streamlit Community Cloud, reboot the app.

Expected dependency log:

```text
scikit-learn==1.9.0
```

The committed model was prepared with Python 3.13. If Community Cloud runs Python
3.14, the application detects the runtime difference from the metadata file and
rebuilds the model from `data/processed/*.csv` before loading it. The rebuild is
automatic and normally happens only after a fresh deployment or runtime change.
