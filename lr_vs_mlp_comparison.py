"""
lr_vs_mlp_comparison.py

Compares Logistic Regression (linear) vs MLP (non-linear) on the same
459 EN-selected features, same train/val split used in Phase 3.

Tests:
  1. LR (hard labels, scaled features)
  2. MLP retrained with hard labels  (matches thesis-reported AUC)
  3. MLP retrained with soft labels  (label ablation)
  4. Per-turn vs trajectory-level FPR at tau=0.4

AUC definition (same as Phase 3 / cell 91):
  - Turn-level ROC-AUC
  - Hard binary label: 1 if judge score > 8, else 0
  - Val set only (100 held-out trajectories, 828 turns)

Usage:
  conda run -n sae python lr_vs_mlp_comparison.py
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ── Config (match Phase 3) ────────────────────────────────────────────
MLP_HIDDEN  = [64, 32]
MLP_DROPOUT = 0.2
MLP_LR      = 1e-3
MLP_EPOCHS  = 100
PATIENCE    = 10
SEED        = 42
TAU         = 0.4

PHASE3_DIR   = Path("C:\\Users\\Lab622_TV\\Documents\\GitHub\\SAE-Jailbreak-Research\\results\\mlp_detector")
DATASET_PATH = PHASE3_DIR / "trajectory_dataset.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── MLP architecture (matches Phase 3) ───────────────────────────────
class DecouplingMLP(nn.Module):
    def __init__(self, input_dim, hidden=MLP_HIDDEN, dropout=MLP_DROPOUT):
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


# ── Load dataset ──────────────────────────────────────────────────────
print("\nLoading dataset...")
saved        = torch.load(DATASET_PATH, weights_only=False, map_location="cpu")
train_dataset = saved["train"]
val_dataset   = saved["val"]
N_FEATURES    = saved["config"]["n_features"]
print(f"  Train: {len(train_dataset)} trajectories")
print(f"  Val:   {len(val_dataset)} trajectories")
print(f"  Features: {N_FEATURES}")


# ── Build flat turn arrays ────────────────────────────────────────────
def build_arrays(dataset):
    X_list, y_hard_list, y_soft_list = [], [], []
    for traj in dataset:
        for feat_vec, score in zip(traj["features"], traj["scores"]):
            X_list.append(np.asarray(feat_vec, dtype=np.float32))
            y_hard_list.append(1 if score > 8 else 0)
            y_soft_list.append(score / 10.0)
    X      = np.stack(X_list)
    y_hard = np.array(y_hard_list, dtype=np.float32)
    y_soft = np.array(y_soft_list, dtype=np.float32)
    return X, y_hard, y_soft

print("\nBuilding turn arrays...")
X_train, y_train_hard, y_train_soft = build_arrays(train_dataset)
X_val,   y_val_hard,   y_val_soft   = build_arrays(val_dataset)
print(f"  Train turns: {len(X_train)} ({int(y_train_hard.sum())} positive)")
print(f"  Val turns:   {len(X_val)}   ({int(y_val_hard.sum())} positive)")


# ── Fix 1: Logistic Regression with scaling ───────────────────────────
print("\n" + "="*60)
print("[1] Logistic Regression (hard labels, scaled features)")
print("="*60)
scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_val_sc      = scaler.transform(X_val)

lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="lbfgs",
    C=1.0,
    random_state=SEED,
)
lr.fit(X_train_sc, y_train_hard)
lr_probs = lr.predict_proba(X_val_sc)[:, 1]
lr_auc   = roc_auc_score(y_val_hard, lr_probs)
print(f"  Val AUC: {lr_auc:.4f}")


# ── Fix 2: Train MLP from scratch, track best val AUC ─────────────────
def train_mlp(X_tr, y_tr, X_v, y_v, label_mode="hard", seed=SEED):
    """
    Train MLP and return (best_val_auc, val_probs_at_best_auc).
    label_mode: 'hard' (1/0) or 'soft' (score/10).
    AUC always evaluated with hard labels.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DecouplingMLP(input_dim=N_FEATURES).to(device)
    opt   = optim.Adam(model.parameters(), lr=MLP_LR)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    X_v_t  = torch.tensor(X_v,  dtype=torch.float32, device=device)
    y_v_t  = torch.tensor(y_v,  dtype=torch.float32, device=device)  # hard labels for AUC

    if label_mode == "hard":
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    else:
        # soft: passed in as y_tr (already score/10)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    best_val_loss  = float("inf")
    best_val_auc   = 0.0
    best_probs     = np.zeros(len(X_v), dtype=np.float32)
    patience_count = 0

    for epoch in range(MLP_EPOCHS):
        # ── Train ──
        model.train()
        perm = torch.randperm(len(X_tr_t))
        preds = model(X_tr_t[perm]).squeeze(-1)
        loss  = F.binary_cross_entropy(preds, y_tr_t[perm])
        opt.zero_grad()
        loss.backward()
        opt.step()

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_preds = model(X_v_t).squeeze(-1)
            val_loss  = F.binary_cross_entropy(val_preds, y_v_t).item()
            val_probs = val_preds.cpu().numpy()

        val_auc = roc_auc_score(y_val_hard, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_probs   = val_probs.copy()

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                break

    return best_val_auc, best_probs


print("\n" + "="*60)
print("[2] MLP retrained — Hard labels (matches thesis config)")
print("="*60)
mlp_hard_auc, mlp_hard_probs = train_mlp(
    X_train, y_train_hard, X_val, y_val_hard, label_mode="hard"
)
print(f"  Best Val AUC: {mlp_hard_auc:.4f}")

print("\n" + "="*60)
print("[3] MLP retrained — Soft labels (label ablation)")
print("="*60)
mlp_soft_auc, mlp_soft_probs = train_mlp(
    X_train, y_train_soft, X_val, y_val_hard, label_mode="soft"
)
print(f"  Best Val AUC: {mlp_soft_auc:.4f}")


# ── Summary: LR vs MLP ───────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY — Linear vs Non-linear (turn-level AUC, hard labels)")
print("="*60)
print(f"  {'Model':<35} {'AUC':>6}")
print(f"  {'-'*42}")
print(f"  {'Logistic Regression (linear)':<35} {lr_auc:.4f}")
print(f"  {'MLP — Hard labels':<35} {mlp_hard_auc:.4f}")
print(f"  {'MLP — Soft labels':<35} {mlp_soft_auc:.4f}")
print(f"  {'-'*42}")
print(f"  Delta (MLP hard - LR)          : {mlp_hard_auc - lr_auc:+.4f}")
print(f"  Delta (MLP hard - MLP soft)    : {mlp_hard_auc - mlp_soft_auc:+.4f}")
print("="*60)
if mlp_hard_auc > lr_auc:
    print("  => Non-linearity HELPS: MLP (hard) outperforms LR")
else:
    print("  => Non-linearity does NOT help: LR >= MLP (hard)")


# ── Per-turn vs Trajectory-level FPR at tau=0.4 (hard-label MLP) ─────
print(f"\n" + "="*60)
print(f"FPR BREAKDOWN AT tau={TAU}  [using hard-label MLP]")
print("="*60)

# Turn-level FPR
safe_mask   = (y_val_hard == 0)
n_safe      = safe_mask.sum()
n_fp_turns  = ((mlp_hard_probs > TAU) & safe_mask).sum()
turn_fpr    = n_fp_turns / n_safe

print(f"\n  Turn-level (per individual turn):")
print(f"    Safe turns          : {n_safe}")
print(f"    Turns with D_t>{TAU} : {n_fp_turns}")
print(f"    Turn-level FPR      : {turn_fpr:.3f}  ({turn_fpr*100:.1f}%)")

# Trajectory-level FPR
turn_idx = 0
traj_fp, traj_tn = 0, 0
for traj in val_dataset:
    n_turns    = len(traj["scores"])
    traj_label = traj["label"]
    traj_probs = mlp_hard_probs[turn_idx : turn_idx + n_turns]
    turn_idx  += n_turns
    if traj_label == 0:
        if (traj_probs > TAU).any():
            traj_fp += 1
        else:
            traj_tn += 1

traj_fpr = traj_fp / (traj_fp + traj_tn)

print(f"\n  Trajectory-level (fires on any turn in trajectory):")
print(f"    Safe trajectories   : {traj_fp + traj_tn}")
print(f"    FP trajectories     : {traj_fp}")
print(f"    TN trajectories     : {traj_tn}")
print(f"    Trajectory-level FPR: {traj_fpr:.3f}  ({traj_fpr*100:.1f}%)")
print(f"    (Expected ~0.508 from Phase 3 cell 77 soft-label model)")

print(f"\n  Summary:")
print(f"    Turn-level FPR       : {turn_fpr:.3f}  <- % of safe turns steered")
print(f"    Trajectory-level FPR : {traj_fpr:.3f}  <- % of safe conversations flagged")
print("="*60)
