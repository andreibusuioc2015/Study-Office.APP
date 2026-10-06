"""The logistic regression without pickles: preprocessing + coefficients as plain numbers (two JSON files).

Same idea as the hotel app's portable.py, but for make_pipeline(ColumnTransformer, LogisticRegression).
These files load with any recent pandas and numpy, on any Python, so the app needs no scikit-learn.

  export(pipe, folder)   -> writes preprocess.json + logit.json   (needs scikit-learn; run in the notebook)
  Model(folder)          -> .transform(df), .predict_proba(df), .contributions(df)   (needs only pandas, numpy)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def export(pipe, folder):
    """Turn a make_pipeline(ColumnTransformer[num: imputer+scaler, cat: one-hot], LogisticRegression) into files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prep, clf = pipe[0], pipe[-1]
    num_pipe, num_cols = prep.named_transformers_["num"], prep.transformers_[0][2]
    imputer, scaler = num_pipe[0], num_pipe[-1]
    onehot, cat_cols = prep.named_transformers_["cat"], prep.transformers_[1][2]
    infrequent = onehot.infrequent_categories_ or [None] * len(cat_cols)
    spec = {
        "numeric": [{"name": c, "median": float(imputer.statistics_[i]), "mean": float(scaler.mean_[i]),
                     "scale": float(scaler.scale_[i])} for i, c in enumerate(num_cols)],
        "categorical": [{"name": c,
                         "frequent": [str(v) for v in onehot.categories_[i] if infrequent[i] is None or v not in infrequent[i]],
                         "has_infrequent": infrequent[i] is not None}
                        for i, c in enumerate(cat_cols)],
        "columns": [n.split("__", 1)[1] for n in prep.get_feature_names_out()],
    }
    weights = {"intercept": float(clf.intercept_[0]),
               "coefficients": [float(w) for w in clf.coef_[0]]}      # same order as spec["columns"]
    (folder / "preprocess.json").write_text(json.dumps(spec, indent=1))
    (folder / "logit.json").write_text(json.dumps(weights, indent=1))


class Model:
    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.loads((folder / "preprocess.json").read_text())
        w = json.loads((folder / "logit.json").read_text())
        self.intercept = w["intercept"]
        self.coef = np.array(w["coefficients"])
        self.features = [n["name"] for n in self.spec["numeric"]] + [c["name"] for c in self.spec["categorical"]]

    def transform(self, df):
        """Same numbers the scikit-learn preprocessing produces: median-fill + standardise; one-hot with rare and
        unseen levels pooled into one column (as OneHotEncoder(handle_unknown='infrequent_if_exist'))."""
        parts = []
        for n in self.spec["numeric"]:
            x = pd.to_numeric(df[n["name"]], errors="coerce").astype(float).fillna(n["median"])
            parts.append(((x - n["mean"]) / n["scale"]).to_numpy()[:, None])
        for c in self.spec["categorical"]:
            v = df[c["name"]].astype(str).to_numpy()
            parts.append(np.stack([v == level for level in c["frequent"]], axis=1).astype(float))
            if c["has_infrequent"]:
                parts.append((~np.isin(v, c["frequent"])).astype(float)[:, None])
        return np.hstack(parts)

    def predict_proba(self, df):
        """Risk of leaving for each row (1-D array): sigmoid(intercept + sum of coefficient x value)."""
        z = self.intercept + self.transform(df) @ self.coef
        return 1 / (1 + np.exp(-z))

    def contributions(self, df):
        """Per-row push of each original feature on the log-odds (+ = towards leaving); one-hot columns summed back."""
        contrib = self.transform(df) * self.coef
        owner = [next(f for f in self.features if col == f or col.startswith(f + "_")) for col in self.spec["columns"]]
        return pd.DataFrame(contrib, columns=owner, index=df.index).T.groupby(level=0).sum().T[self.features]
