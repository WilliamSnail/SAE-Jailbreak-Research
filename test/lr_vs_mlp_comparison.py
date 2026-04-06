"""
lr_vs_mlp_comparison.py

Compares Logistic Regression (linear) vs MLP (non-linear) on the same
459 EN-selected features, same train/val split used in Phase 3.

Purpose: test whether non-linearity actually helps, to support the
H1 claim that the joint F_H/F_S condition is XOR-like.

AUC definition (same as Phase 3 / cell 91):
  - Turn-level ROC-AUC
  - Hard binary label: 1 if judge score > 8, else 0
  - Val set only (100 held-out trajectories, 828 turns)

Usage:
  conda run -n sae python test/lr_vs_mlp_comparison.py
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Paths ────────────────────────────────────────────────────────────
PHASE3_DIR = Path("results/mlp_detector")
DATASET_PATH = PHASE3_DIR / "trajectory_dataset.pt"
MODEL_PATH   = PHASE3_DIR / "best_model.pt"

# ── MLP architecture (must match Phase 3 training) ───────────────────
class DecouplingMLP(nn.Module):
    def __init__(self, input_dim, hidden=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


# ── Load dataset ─────────────────────────────────────────────────────
print("Loading dataset...")
saved = torch.load(DATASET_PATH, weights_only=False, map_location="cpu")
train_dataset = saved["train"]
val_dataset   = saved["val"]
N_FEATURES    = saved["config"]["n_features"]
print(f"  Train: {len(train_dataset)} trajectories")
print(f"  Val:   {len(val_dataset)} trajectories")
print(f"  Features: {N_FEATURES}")


# ── Build flat turn arrays ────────────────────────────────────────────
def build_arrays(dataset):
    X_list, y_list = [], []
    for traj in dataset:
        for feat_vec, score in zip(traj["features"], traj["scores"]):
            X_list.append(np.asarray(feat_vec, dtype=np.float32))
            y_list.append(1 if score > 8 else 0)
    return np.stack(X_list), np.array(y_list)

print("\nBuilding turn arrays...")
X_train, y_train = build_arrays(train_dataset)
X_val,   y_val   = build_arrays(val_dataset)
print(f"  Train turns: {len(X_train)} ({y_train.sum()} positive)")
print(f"  Val turns:   {len(X_val)}   ({y_val.sum()} positive)")


# ── Logistic Regression ───────────────────────────────────────────────
print("\n[1] Logistic Regression (linear, same 459 features)...")
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    C=1.0,
    random_state=42,
)
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_val)[:, 1]
lr_auc   = roc_auc_score(y_val, lr_probs)
print(f"  Val AUC: {lr_auc:.4f}")


# ── MLP (load trained weights from Phase 3) ───────────────────────────
print("\n[2] MLP (non-linear, same 459 features, Phase 3 weights)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = DecouplingMLP(input_dim=N_FEATURES).to(device)
checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=device)
mlp.load_state_dict(checkpoint["model_state"])
mlp.eval()
print(f"  Loaded epoch {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.4f}")

mlp_probs = []
with torch.no_grad():
    for start in range(0, len(X_val), 512):
        batch = torch.tensor(X_val[start:start+512], dtype=torch.float32, device=device)
        out = mlp(batch).squeeze(-1).cpu().numpy()
        mlp_probs.extend(out.tolist())
mlp_probs = np.array(mlp_probs)
mlp_auc   = roc_auc_score(y_val, mlp_probs)
print(f"  Val AUC: {mlp_auc:.4f}")


# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print("COMPARISON (same 459 features, same val set, turn-level AUC)")
print("="*55)
print(f"  Logistic Regression (linear) : {lr_auc:.4f}")
print(f"  MLP (non-linear)             : {mlp_auc:.4f}")
print(f"  Delta AUC                    : {mlp_auc - lr_auc:+.4f}")
print("="*55)
if mlp_auc > lr_auc:
    print("  => Non-linearity HELPS: MLP outperforms linear classifier")
else:
    print("  => Non-linearity does NOT help on this data")
