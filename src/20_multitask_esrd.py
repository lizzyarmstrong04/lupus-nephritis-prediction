"""
Multi-task neural network: ESRD 5-year + ESRD 10-year jointly.
Same 796 patients, shared trunk, two task heads.
Compared against single-task MLP and best classical models.
"""

import warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

BASE = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"

# Load & merge ESRD datasets (same 796 patients)
df5  = pd.read_excel(f"{BASE}/Data/Processed/esrd_5yr_selected.xlsx")
df10 = pd.read_excel(f"{BASE}/Data/Processed/esrd_10yr_selected.xlsx")

y5  = df5["esrd_5yr"].astype(int)
y10 = df10["esrd_10yr"].astype(int)

X5  = df5.drop(columns=["esrd_5yr"])
X10 = df10.drop(columns=["esrd_10yr"])

# Union of all features (same patients so column-merge is safe)
X_all = pd.concat([X5, X10], axis=1).loc[:, ~pd.concat([X5, X10], axis=1).columns.duplicated()]
X_all.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in X_all.columns]

print(f"Merged feature matrix: {X_all.shape}  ({X_all.shape[1]} features)")
print(f"Events — ESRD 5yr: {y5.sum()} ({y5.mean()*100:.1f}%)  "
      f"ESRD 10yr: {y10.sum()} ({y10.mean()*100:.1f}%)")

# Multi-task network
class MultiTaskNet(nn.Module):
    def __init__(self, n_features, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head5  = nn.Linear(in_dim, 1)
        self.head10 = nn.Linear(in_dim, 1)

    def forward(self, x):
        z = self.trunk(x)
        return torch.sigmoid(self.head5(z)).squeeze(1), \
               torch.sigmoid(self.head10(z)).squeeze(1)

# Single-task network (same architecture, one head)
class SingleTaskNet(nn.Module):
    def __init__(self, n_features, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Linear(in_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.head(self.trunk(x))).squeeze(1)

# Training helpers
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

def train_multitask(X_tr, y5_tr, y10_tr, n_features,
                    epochs=150, lr=1e-3, batch=64):
    model = MultiTaskNet(n_features)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce   = nn.BCELoss()
    ds    = TensorDataset(to_tensor(X_tr),
                          to_tensor(y5_tr), to_tensor(y10_tr))
    dl    = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, y5b, y10b in dl:
            p5, p10 = model(xb)
            loss = bce(p5, y5b) + bce(p10, y10b)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_singletask(X_tr, y_tr, n_features, epochs=150, lr=1e-3, batch=64):
    model = SingleTaskNet(n_features)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce   = nn.BCELoss()
    ds    = TensorDataset(to_tensor(X_tr), to_tensor(y_tr))
    dl    = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            loss = bce(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def predict(model, X_te):
    model.eval()
    with torch.no_grad():
        out = model(to_tensor(X_te))
    if isinstance(out, tuple):
        return out[0].numpy(), out[1].numpy()
    return out.numpy()

# CV evaluation
def run_cv(X, y5, y10, n_splits=10, n_repeats=5):
    CV = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=42)
    n_features = X.shape[1]

    mt_aucs5, mt_aucs10 = [], []
    st_aucs5, st_aucs10 = [], []

    for fold_i, (tr, te) in enumerate(CV.split(X, y5)):   # stratify on 5yr
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr])
        X_te   = scaler.transform(X[te])
        y5_tr  = y5[tr].astype(np.float32)
        y10_tr = y10[tr].astype(np.float32)

        # Multi-task
        mt = train_multitask(X_tr, y5_tr, y10_tr, n_features)
        p5_mt, p10_mt = predict(mt, X_te)
        mt_aucs5.append(roc_auc_score(y5[te], p5_mt))
        mt_aucs10.append(roc_auc_score(y10[te], p10_mt))

        # Single-task 5yr
        st5 = train_singletask(X_tr, y5_tr, n_features)
        st_aucs5.append(roc_auc_score(y5[te], predict(st5, X_te)))

        # Single-task 10yr
        st10 = train_singletask(X_tr, y10_tr, n_features)
        st_aucs10.append(roc_auc_score(y10[te], predict(st10, X_te)))

        if (fold_i + 1) % 10 == 0:
            print(f"  fold {fold_i+1}/50 | "
                  f"MT 5yr={np.mean(mt_aucs5):.3f}  "
                  f"MT 10yr={np.mean(mt_aucs10):.3f}  "
                  f"ST 5yr={np.mean(st_aucs5):.3f}  "
                  f"ST 10yr={np.mean(st_aucs10):.3f}", flush=True)

    def ci(aucs):
        return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

    return {
        "Multi-task — ESRD 5yr":  ci(mt_aucs5),
        "Multi-task — ESRD 10yr": ci(mt_aucs10),
        "Single-task — ESRD 5yr": ci(st_aucs5),
        "Single-task — ESRD 10yr":ci(st_aucs10),
    }

# Run
print("\nRunning 5×10-fold CV (50 folds) — this takes ~10 min...\n")
results = run_cv(X_all.values, y5.values, y10.values)

print("\n" + "="*65)
print("RESULTS")
print("="*65)
rows = []
for label, (mean, lo, hi) in results.items():
    print(f"  {label:35s}  AUROC={mean:.3f} [{lo:.3f}–{hi:.3f}]")
    rows.append({"Model": label, "AUROC": round(mean,3),
                 "95% CI lower": round(lo,3), "95% CI upper": round(hi,3)})

# Add best classical models for reference
classical = {
    "LR (classical) — ESRD 5yr":  (0.797, 0.669, 0.904),
    "LR (classical) — ESRD 10yr": (0.811, 0.656, 0.903),
    "XGB (classical) — ESRD 5yr": (0.792, 0.631, 0.901),
    "XGB (classical) — ESRD 10yr":(0.821, 0.696, 0.933),
}
print("\n  --- Classical model reference ---")
for label, (mean, lo, hi) in classical.items():
    print(f"  {label:35s}  AUROC={mean:.3f} [{lo:.3f}–{hi:.3f}]")
    rows.append({"Model": label, "AUROC": round(mean,3),
                 "95% CI lower": round(lo,3), "95% CI upper": round(hi,3)})

# Save
out = f"{BASE}/outputs/multitask_esrd_results.xlsx"
pd.DataFrame(rows).to_excel(out, index=False)
print(f"\nSaved: {out}")
