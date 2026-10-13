"""
lr_vs_mlp_comparison.py

Full 2x2 ablation: loss_mode x label_mode, plus LR baseline and EWL.

Variants trained:
  Standard BCE  x Soft labels  (deployed model, matches Phase 3 cell 75)
  Standard BCE  x Hard labels
  Softmax BCE   x Soft labels  (matches Phase 3 cell 82 ablation)
  Softmax BCE   x Hard labels  (best AUC in cell 82)

Plus:
  LR — hard labels, StandardScaler, max_iter=5000
  EWL sweep at tau in {0.3, 0.4, 0.5, 0.6} for all 4 MLP variants
  Per-turn vs trajectory-level FPR at tau=0.4 (deployed model)

Training loop matches Phase 3 cell 75 exactly:
  - Per-trajectory (not batched)
  - Adam lr=1e-3, epochs=100, patience=10 on val loss
  - Val AUC always with hard labels (score > 8), matching cell 75

SoftmaxWeightedBCE matches Phase 3 cell 72:
  weights = softmax(raw_scores), loss = sum(w * BCE(pred, target))

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

# ── Config (must match Phase 3) ───────────────────────────────────────
MLP_HIDDEN  = [64, 32]
MLP_DROPOUT = 0.2
MLP_LR      = 1e-3
MLP_EPOCHS  = 100
PATIENCE    = 10
SEED             = 42
ABLATION_SEEDS   = [42, 123, 456, 789, 1024]   # matches Phase 3 cell 82
TAU_SWEEP        = [0.3, 0.4, 0.5, 0.6]
TAU_FPR     = 0.4   # threshold used for FPR breakdown

PHASE3_DIR   = Path("C:\\Users\\Lab622_TV\\Documents\\GitHub\\SAE-Jailbreak-Research\\results\\mlp_detector")
DATASET_PATH = PHASE3_DIR / "trajectory_dataset.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── SoftmaxWeightedBCE (matches Phase 3 cell 72) ─────────────────────
class SoftmaxWeightedBCE(nn.Module):
    """
    Weights each turn by softmax(raw_scores) so high-score turns
    dominate the gradient. Matches cell 72 exactly.
    """
    def forward(self, predictions, targets, raw_scores):
        weights      = F.softmax(raw_scores, dim=0)
        per_turn_bce = F.binary_cross_entropy(predictions, targets, reduction="none")
        return (weights * per_turn_bce).sum()


# ── MLP architecture (matches Phase 3 cell 71) ───────────────────────
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
saved         = torch.load(DATASET_PATH, weights_only=False, map_location="cpu")
train_dataset = saved["train"]
val_dataset   = saved["val"]
N_FEATURES    = saved["config"]["n_features"]
print(f"  Train: {len(train_dataset)} trajectories")
print(f"  Val:   {len(val_dataset)} trajectories")
print(f"  Features: {N_FEATURES}")


# ── Build flat turn arrays (for LR and FPR) ───────────────────────────
def build_flat_arrays(dataset):
    """Returns X (n_turns, N_FEATURES), y_hard, y_soft as flat arrays."""
    X_list, yh_list, ys_list = [], [], []
    for traj in dataset:
        for feat_vec, score in zip(traj["features"], traj["scores"]):
            X_list.append(np.asarray(feat_vec, dtype=np.float32))
            yh_list.append(1 if score > 8 else 0)
            ys_list.append(score / 10.0)
    return (np.stack(X_list),
            np.array(yh_list, dtype=np.float32),
            np.array(ys_list, dtype=np.float32))

print("\nBuilding turn arrays...")
X_train, y_train_hard, y_train_soft = build_flat_arrays(train_dataset)
X_val,   y_val_hard,   y_val_soft   = build_flat_arrays(val_dataset)
print(f"  Train turns: {len(X_train)} ({int(y_train_hard.sum())} positive)")
print(f"  Val turns:   {len(X_val)}   ({int(y_val_hard.sum())} positive)")


# ─────────────────────────────────────────────────────────────────────
# [1] Logistic Regression — hard labels, scaled (Fix 1)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[1] Logistic Regression (hard labels, StandardScaler, max_iter=5000)")
print("="*60)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

lr_clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="lbfgs",
    C=1.0,
    random_state=SEED,
)
lr_clf.fit(X_train_sc, y_train_hard)
lr_probs = lr_clf.predict_proba(X_val_sc)[:, 1]
lr_auc   = roc_auc_score(y_val_hard, lr_probs)
print(f"  Val AUC: {lr_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────
# MLP training function — matches Phase 3 cells 75 & 82 exactly
# ─────────────────────────────────────────────────────────────────────
softmax_bce_fn = SoftmaxWeightedBCE()

def train_mlp(train_data, val_data, label_mode="soft", loss_mode="standard", seed=SEED):
    """
    Train MLP per-trajectory.

    label_mode : 'soft' (score/10) | 'hard' (1 if score>8 else 0)
    loss_mode  : 'standard' (uniform BCE) | 'softmax' (SoftmaxWeightedBCE)

    Val loss uses same label_mode and loss_mode as training.
    Val AUC always uses hard labels (score > 8), matching Phase 3 cell 75.

    Returns:
      best_val_auc   — highest val AUC seen during training
      best_val_probs — val predictions at best-AUC epoch
      loss_val_probs — val predictions at best-val-loss epoch
                       (equivalent to what best_model.pt saves)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = DecouplingMLP(input_dim=N_FEATURES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR)

    n_val_turns    = sum(len(t["scores"]) for t in val_data)
    best_val_loss  = float("inf")
    best_val_auc   = 0.0
    best_val_probs = np.zeros(n_val_turns, dtype=np.float32)
    loss_val_probs = np.zeros(n_val_turns, dtype=np.float32)
    patience_count = 0

    for epoch in range(MLP_EPOCHS):
        # ── Train: per-trajectory (matches cell 75) ───────────────────
        model.train()
        train_losses = []
        perm = np.random.permutation(len(train_data))

        for idx in perm:
            traj       = train_data[idx]
            features   = traj["features"]
            scores_raw = traj["scores"]

            if len(features) < 2:
                continue

            X_traj = torch.tensor(
                np.stack(features), dtype=torch.float32, device=device
            )
            scores_t = torch.tensor(
                scores_raw, dtype=torch.float32, device=device
            )

            if label_mode == "soft":
                targets = scores_t / 10.0
            else:
                targets = (scores_t > 8).float()

            preds = model(X_traj).squeeze(-1)

            if loss_mode == "softmax":
                loss = softmax_bce_fn(preds, targets, scores_t)
            else:
                loss = F.binary_cross_entropy(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_losses   = []
        all_preds    = []
        all_hard_tgt = []

        with torch.no_grad():
            for traj in val_data:
                features   = traj["features"]
                scores_raw = traj["scores"]

                if len(features) < 2:
                    continue

                X_traj   = torch.tensor(
                    np.stack(features), dtype=torch.float32, device=device
                )
                scores_t = torch.tensor(
                    scores_raw, dtype=torch.float32, device=device
                )

                if label_mode == "soft":
                    val_targets = scores_t / 10.0
                else:
                    val_targets = (scores_t > 8).float()

                preds = model(X_traj).squeeze(-1)

                if loss_mode == "softmax":
                    loss = softmax_bce_fn(preds, val_targets, scores_t)
                else:
                    loss = F.binary_cross_entropy(preds, val_targets)

                val_losses.append(loss.item())
                all_preds.extend(preds.cpu().numpy())
                # AUC always hard labels (matches cell 75: targets > 0.8)
                all_hard_tgt.extend([1 if s > 8 else 0 for s in scores_raw])

        avg_val_loss = np.mean(val_losses)
        preds_np     = np.array(all_preds)
        hard_tgt_np  = np.array(all_hard_tgt)

        val_auc = (roc_auc_score(hard_tgt_np, preds_np)
                   if len(np.unique(hard_tgt_np)) > 1 else 0.0)

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_val_probs = preds_np.copy()

        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            loss_val_probs = preds_np.copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    Early stop ep {epoch+1} "
                      f"(best_loss={best_val_loss:.4f}, best_AUC={best_val_auc:.4f})")
                break

    return best_val_auc, best_val_probs, loss_val_probs


# ─────────────────────────────────────────────────────────────────────
# [2–5] MLP 2×2 ablation — 5 seeds each (matches Phase 3 ABLATION_SEEDS)
# ─────────────────────────────────────────────────────────────────────
VARIANTS = [
    ("Standard BCE × Soft (deployed)", "soft", "standard"),
    ("Standard BCE × Hard",            "hard", "standard"),
    ("Softmax BCE  × Soft",            "soft", "softmax"),
    ("Softmax BCE  × Hard",            "hard", "softmax"),
]

# Results store: variant_name → list of (auc, best_probs) per seed
variant_results = {}   # name → {"aucs": [], "probs_per_seed": []}

for v_name, lbl, loss in VARIANTS:
    print("\n" + "="*60)
    print(f"MLP — {v_name}  (5 seeds)")
    print("="*60)
    aucs, probs_list = [], []
    for s in ABLATION_SEEDS:
        auc, probs_bestauc, _ = train_mlp(
            train_dataset, val_dataset,
            label_mode=lbl, loss_mode=loss, seed=s
        )
        aucs.append(auc)
        probs_list.append(probs_bestauc)
        print(f"  seed={s:4d}  AUC={auc:.4f}")
    print(f"  Mean AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    variant_results[v_name] = {"aucs": aucs, "probs_per_seed": probs_list,
                                "label_mode": lbl, "loss_mode": loss}


# ─────────────────────────────────────────────────────────────────────
# Summary: LR vs all MLP variants — AUC (mean ± std)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY — 2×2 MLP ablation + LR (turn-level AUC, hard label eval)")
print("="*60)
print(f"  {'Model':<44} {'AUC (mean ± std)':>18}")
print(f"  {'-'*63}")
print(f"  {'Logistic Regression (hard labels, scaled)':<44} {'':>9}{lr_auc:.4f}")
print(f"  {'-'*63}")
for v_name, res in variant_results.items():
    aucs = res["aucs"]
    print(f"  {v_name:<44} {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print("="*60)


# ─────────────────────────────────────────────────────────────────────
# EWL evaluation function — matches Phase 3 cell 77
# Returns dict: tau → {prec, rec, f1, fpr, ew_mean, ew_med, tp, fp, fn, tn}
# ─────────────────────────────────────────────────────────────────────
def compute_ewl_metrics(val_data, val_probs, tau_list=TAU_SWEEP,
                        y_hard=None):
    """
    Compute detection metrics + EWL for each tau.
    val_probs : flat array of MLP outputs for all val turns (in order).
    y_hard    : flat array of hard labels (0/1) for turn-level FPR.
                If None, turn-level FPR is not computed.
    Matches Phase 3 cell 77 logic exactly.
    Returns dict keyed by tau.
    """
    traj_probs = []
    turn_idx = 0
    for traj in val_data:
        n = len(traj["scores"])
        traj_probs.append({
            "probs":   val_probs[turn_idx : turn_idx + n],
            "scores":  traj["scores"],
            "label":   traj["label"],
            "jb_turn": next((i for i, s in enumerate(traj["scores"]) if s > 8), None),
        })
        turn_idx += n

    results = {}
    for tau in tau_list:
        tp = fp = fn = tn = 0
        latencies = []
        for ex in traj_probs:
            det_turn = next((i for i, p in enumerate(ex["probs"]) if p > tau), None)
            if ex["label"] == 1:
                if det_turn is not None:
                    tp += 1
                    if ex["jb_turn"] is not None:
                        latencies.append(ex["jb_turn"] - det_turn)
                else:
                    fn += 1
            else:
                if det_turn is not None:
                    fp += 1
                else:
                    tn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        traj_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Turn-level FPR: % of individual safe turns that exceed tau
        if y_hard is not None:
            safe_mask   = (y_hard == 0)
            n_safe      = int(safe_mask.sum())
            n_fp_turns  = int(((val_probs > tau) & safe_mask).sum())
            turn_fpr    = n_fp_turns / n_safe if n_safe > 0 else 0.0
        else:
            turn_fpr = float("nan")

        results[tau] = dict(
            prec=prec, rec=rec, f1=f1,
            traj_fpr=traj_fpr, turn_fpr=turn_fpr,
            ew_mean=(np.mean(latencies)   if latencies else float("nan")),
            ew_med= (np.median(latencies) if latencies else float("nan")),
            tp=tp, fp=fp, fn=fn, tn=tn,
        )
    return results


def print_ewl_table(metrics_by_tau, tau_list=TAU_SWEEP):
    print(f"\n  {'tau':>4}  {'P':>6}  {'R':>6}  {'F1':>6}  {'FPR':>6}  "
          f"{'EW_mean':>8}  {'EW_med':>7}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  "
          f"{'─'*8}  {'─'*7}  {'─'*3} {'─'*3} {'─'*3} {'─'*3}")
    for tau in tau_list:
        m = metrics_by_tau[tau]
        print(f"  {tau:4.1f}  {m['prec']:6.3f}  {m['rec']:6.3f}  {m['f1']:6.3f}  "
              f"{m['traj_fpr']:6.3f}  {m['ew_mean']:+8.1f}  {m['ew_med']:+7.1f}  "
              f"{m['tp']:3d} {m['fp']:3d} {m['fn']:3d} {m['tn']:3d}")


# ─────────────────────────────────────────────────────────────────────
# EWL: 5-seed evaluation for all 4 MLP variants
# Reports mean ± std of EW_mean and EW_med across seeds at each tau
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EWL — 5-seed mean ± std across all variants")
print("="*60)

# Per-seed EWL table for the deployed model (Standard+Soft) — for
# comparison against Phase 3 cell 77
deployed_name = "Standard BCE × Soft (deployed)"
print(f"\n--- Per-seed EWL: {deployed_name} ---")
for s, probs in zip(ABLATION_SEEDS, variant_results[deployed_name]["probs_per_seed"]):
    print(f"  seed={s}")
    m = compute_ewl_metrics(val_dataset, probs, y_hard=y_val_hard)
    print_ewl_table(m)

# Aggregate: for each variant, collect metrics per tau across seeds
print("\n--- 5-seed aggregate EWL (mean ± std) ---")
for v_name, res in variant_results.items():
    print(f"\n  {v_name}")
    seed_metrics = [
        compute_ewl_metrics(val_dataset, probs, y_hard=y_val_hard)
        for probs in res["probs_per_seed"]
    ]
    print(f"  {'tau':>4}  {'EWL_med (mean±std)':>20}  "
          f"{'Traj-FPR (mean±std)':>22}  {'Turn-FPR (mean±std)':>22}")
    print(f"  {'─'*4}  {'─'*20}  {'─'*22}  {'─'*22}")
    for tau in TAU_SWEEP:
        ew_meds    = [sm[tau]["ew_med"]    for sm in seed_metrics]
        traj_fprs  = [sm[tau]["traj_fpr"]  for sm in seed_metrics]
        turn_fprs  = [sm[tau]["turn_fpr"]  for sm in seed_metrics]
        ew_meds_clean = [x for x in ew_meds if not np.isnan(x)]
        m_med      = (f"{np.mean(ew_meds_clean):+.2f}±{np.std(ew_meds_clean):.2f}"
                      if ew_meds_clean else "nan")
        m_traj_fpr = f"{np.mean(traj_fprs):.3f}±{np.std(traj_fprs):.3f}"
        m_turn_fpr = f"{np.mean(turn_fprs):.3f}±{np.std(turn_fprs):.3f}"
        print(f"  {tau:4.1f}  {m_med:>20}  {m_traj_fpr:>22}  {m_turn_fpr:>22}")


# ─────────────────────────────────────────────────────────────────────
# EWL: Logistic Regression baseline
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EWL — Logistic Regression baseline")
print("="*60)

lr_ewl_metrics = compute_ewl_metrics(val_dataset, lr_probs, y_hard=y_val_hard)
print_ewl_table(lr_ewl_metrics)

print(f"\n  {'tau':>4}  {'EWL_med':>10}  {'Traj-FPR':>10}  {'Turn-FPR':>10}")
print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}")
for tau in TAU_SWEEP:
    m = lr_ewl_metrics[tau]
    print(f"  {tau:4.1f}  {m['ew_med']:+10.2f}  {m['traj_fpr']:10.3f}  {m['turn_fpr']:10.3f}")


# ─────────────────────────────────────────────────────────────────────
# FPR breakdown at tau=0.4 — Standard+Soft seed=42 (for reference)
# ─────────────────────────────────────────────────────────────────────
print(f"\n" + "="*60)
print(f"FPR BREAKDOWN AT tau={TAU_FPR}  [Standard BCE × Soft, seed=42]")
print(f"  (Reference: deployed model used in Phase 4 & 5 is best_model.pt)")
print("="*60)

probs = variant_results["Standard BCE × Soft (deployed)"]["probs_per_seed"][0]  # seed=42

# Turn-level
safe_mask  = (y_val_hard == 0)
n_safe     = int(safe_mask.sum())
n_fp_turns = int(((probs > TAU_FPR) & safe_mask).sum())
turn_fpr   = n_fp_turns / n_safe

print(f"\n  Turn-level (per individual turn):")
print(f"    Safe turns           : {n_safe}")
print(f"    Turns with D_t>{TAU_FPR} : {n_fp_turns}")
print(f"    Turn-level FPR       : {turn_fpr:.3f}  ({turn_fpr*100:.1f}%)")

# Trajectory-level
turn_idx = 0
traj_fp, traj_tn = 0, 0
for traj in val_dataset:
    n      = len(traj["scores"])
    t_prob = probs[turn_idx : turn_idx + n]
    turn_idx += n
    if traj["label"] == 0:
        if (t_prob > TAU_FPR).any():
            traj_fp += 1
        else:
            traj_tn += 1

traj_fpr = traj_fp / (traj_fp + traj_tn)
print(f"\n  Trajectory-level (fires on any turn):")
print(f"    Safe trajectories    : {traj_fp + traj_tn}")
print(f"    FP trajectories      : {traj_fp}")
print(f"    TN trajectories      : {traj_tn}")
print(f"    Trajectory-level FPR : {traj_fpr:.3f}  ({traj_fpr*100:.1f}%)")
print(f"    (Expected ~0.508 from Phase 3 cell 77)")

print(f"\n  Summary:")
print(f"    Turn-level FPR       : {turn_fpr:.3f}  <- % of safe turns steered")
print(f"    Trajectory-level FPR : {traj_fpr:.3f}  <- % of safe conversations flagged")
print("="*60)
