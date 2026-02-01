#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kerta_dnn_sigmoid_oriented.py

Perbaikan DNN regresi sigmoid untuk keterbacaan kode:
- Memastikan ORIENTASI per-kategori dipatuhi dengan target kategori "teacher deterministik"
  yang dihitung dari fitur yang sudah dinormalisasi orientasi.
- Head Global diawasi ganda: ke label (y/2) dan regularisasi konsistensi agar ~mean(5 kategori).
- Menyediakan kalibrasi Isotonic per-head.

CLI ringkas:
  TRAIN:
    python kerta_dnn_sigmoid_oriented.py train \
      --data data.csv --label-col target \
      --schema schema_kerta.csv \
      --artifact-dir artifacts_kerta --epochs 200 --patience 25

  INFER:
    python kerta_dnn_sigmoid_oriented.py infer \
      --data data_infer.csv \
      --artifact-dir artifacts_kerta \
      --out scores.csv

Catatan akademik:
- Orientasi Higher: makin besar makin readable → normalisasi kuantil + clipping ke [0,1].
- Orientasi Lower: makin kecil makin readable → normalisasi Higher lalu dibalik: 1 - norm.
- Orientasi Mid: sweet-spot di sekitar μ* (rata-rata kelas readable bila ada; fallback μ seluruh data):
    skor = 1 - |x - μ*| / (IQR/2)  (di-clip ke [0,1]).
- Agregasi kategori default mean (bisa median/harmonic).
- Konsistensi: L_cons = Huber( Global, mean(Category) ).
"""

import os
import json
import math
import argparse
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------
# Util IO & Skema
# ------------------------------
def load_schema(schema_csv: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    Membaca CSV skema dengan kolom: feature,category,orientation
    return:
      orientation_map: {feature -> "Higher"/"Lower"/"Mid"}
      category_map   : {feature -> "VisualClarity"/...}
      CATEGORIES     : {category -> [features]}
    """
    df = pd.read_csv(schema_csv)
    req = {"feature", "category", "orientation"}
    if not req.issubset(set(c.lower() for c in df.columns)):
        # toleransi kapitalisasi
        cols = {c.lower(): c for c in df.columns}
        feature_col = cols.get("feature", None)
        category_col = cols.get("category", None)
        orientation_col = cols.get("orientation", None)
    else:
        feature_col, category_col, orientation_col = "feature", "category", "orientation"

    if feature_col is None or category_col is None or orientation_col is None:
        raise ValueError("Schema CSV harus memiliki kolom: feature, category, orientation")

    orientation_map = {}
    category_map = {}
    CATEGORIES: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        f = str(r[feature_col]).strip()
        c = str(r[category_col]).strip()
        o = str(r[orientation_col]).strip()
        if not f:
            continue
        orientation_map[f] = o
        category_map[f] = c
        CATEGORIES.setdefault(c, []).append(f)
    return orientation_map, category_map, CATEGORIES


# ------------------------------
# Normalisasi berorientasi
# ------------------------------
def _quantile_norm(arr: np.ndarray, q_low=0.05, q_high=0.95) -> np.ndarray:
    lo = np.nanquantile(arr, q_low)
    hi = np.nanquantile(arr, q_high)
    if hi <= lo:
        hi = lo + 1e-9
    z = (arr - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0)

def _mid_score(arr: np.ndarray, mu_star: float, iqr: float) -> np.ndarray:
    # skor mid: 1 - jarak relatif terhadap μ*, dinormalisasi dengan IQR/2
    eps = 1e-9
    scale = max(iqr / 2.0, eps)
    s = 1.0 - np.abs(arr - mu_star) / scale
    return np.clip(s, 0.0, 1.0)

def fit_normalizer(df: pd.DataFrame,
                   label_col: str,
                   orientation_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Menyimpan parameter normalisasi untuk setiap fitur sesuai orientasinya.
    Untuk Mid, μ* diambil dari subset label==2 (readable) bila ada; fallback μ seluruh data.
    """
    params: Dict[str, Any] = {"label_col": label_col, "features": [], "per_feature": {}}
    y = df[label_col].values
    # data subset readable = 2
    mask_r = (y == 2)
    for feat, orient in orientation_map.items():
        if feat not in df.columns:
            continue
        x = df[feat].astype(float).values
        orient = orient.strip().lower()
        rec = {"orient": orient}
        if orient in ("higher", "lower"):
            lo = float(np.nanquantile(x, 0.05)) # type: ignore
            hi = float(np.nanquantile(x, 0.95)) # type: ignore
            if hi <= lo:
                hi = lo + 1e-9
            rec.update({"lo": lo, "hi": hi}) # type: ignore
        elif orient == "mid":
            if mask_r.sum() >= 10: # type: ignore
                mu_star = float(np.nanmean(df.loc[mask_r, feat].astype(float).values)) # type: ignore
                iqr = float(np.nanquantile(df.loc[mask_r, feat].values, 0.75) - # type: ignore
                            np.nanquantile(df.loc[mask_r, feat].values, 0.25)) # type: ignore
            else:
                mu_star = float(np.nanmean(x)) # type: ignore
                iqr = float(np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25)) # type: ignore
            if iqr <= 1e-9:
                iqr = 1.0
            rec.update({"mu_star": mu_star, "iqr": iqr}) # type: ignore
        else:
            raise ValueError(f"Orientasi tidak dikenal untuk fitur '{feat}': {orient}")
        params["features"].append(feat)
        params["per_feature"][feat] = rec
    return params

def apply_normalizer(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for feat in params["features"]:
        if feat not in out.columns:
            continue
        rec = params["per_feature"][feat]
        orient = rec["orient"]
        x = out[feat].astype(float).values
        if orient in ("higher", "lower"):
            lo, hi = rec["lo"], rec["hi"]
            if hi <= lo:
                hi = lo + 1e-9
            z = (x - lo) / (hi - lo)
            z = np.clip(z, 0.0, 1.0)
            if orient == "lower":
                z = 1.0 - z
        elif orient == "mid":
            z = _mid_score(x, rec["mu_star"], rec["iqr"])
        else:
            raise ValueError(f"Orientasi tidak dikenal: {orient}")
        out[feat] = z
    return out


# ------------------------------
# Agregasi skor kategori (teacher)
# ------------------------------
def aggregate_category_scores(
    df_norm: pd.DataFrame,
    CATEGORIES: Dict[str, List[str]],
    agg: str = "mean",
    per_metric_weights: Dict[str, Dict[str, float]] | None = None
) -> pd.DataFrame:
    """
    Menghasilkan skor [0..1] per kategori dari fitur yang sudah dinormalisasi orientasi.
    agg: "mean" | "median" | "harmonic"
    """
    import numpy as np

    def _agg(vals: np.ndarray, kind: str) -> float:
        if kind == "median":
            return float(np.median(vals))
        if kind == "harmonic":
            eps = 1e-12
            return float(len(vals) / np.sum(1.0 / np.clip(vals, eps, 1.0)))
        return float(np.mean(vals))  # default mean

    out = {}
    for cat, feats in CATEGORIES.items():
        vals = []
        wts  = []
        for f in feats:
            if f not in df_norm.columns:
                continue
            v = df_norm[f].astype(float).values
            w = 1.0
            if per_metric_weights and cat in per_metric_weights and f in per_metric_weights[cat]:
                w = float(per_metric_weights[cat][f])
            vals.append(v)
            wts.append(w)
        if not vals:
            out[cat] = np.zeros(len(df_norm), dtype=float)
            continue
        V = np.vstack(vals)  # (m, n)
        if agg == "mean":
            W = np.array(wts, dtype=float).reshape(-1, 1)  # (m,1)
            s = W.sum()
            if s <= 0:
                W[:] = 1.0
                s = W.sum()
            W = W / s
            cat_score = (V * W).sum(axis=0)
        else:
            cat_score = np.array([_agg(V[:, i], agg) for i in range(V.shape[1])])
        out[cat] = cat_score
    return pd.DataFrame(out, index=df_norm.index)


# ------------------------------
# Model
# ------------------------------
class MLPOriented(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        def head():
            return nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        # 5 kategori + 1 global
        self.heads = nn.ModuleList([head() for _ in range(6)])

    def forward(self, x):
        h = self.backbone(x)
        outs = [hd(h) for hd in self.heads]  # list of (B,1)
        return outs  # [vis,str,doc,name,cog,glob]


# ------------------------------
# Loss & Train
# ------------------------------
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0):
    return nn.functional.smooth_l1_loss(pred, target, beta=delta, reduction="none")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fold(X_tr, y_tr_cat, y_tr_glb, X_va, y_va_cat, y_va_glb,
               epochs=200, batch_size=32, lr=1e-3, weight_decay=1e-4,
               patience=25, seed=1234, alpha_glob=1.0, beta_cons=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = X_tr.shape[1]
    net = MLPOriented(in_dim).to(DEVICE)
    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Xva = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
    ytr_cat = torch.tensor(y_tr_cat, dtype=torch.float32).to(DEVICE)  # (n,5)
    ytr_glb = torch.tensor(y_tr_glb, dtype=torch.float32).to(DEVICE)  # (n,1)
    yva_cat = torch.tensor(y_va_cat, dtype=torch.float32).to(DEVICE)
    yva_glb = torch.tensor(y_va_glb, dtype=torch.float32).to(DEVICE)

    n = Xtr.shape[0]
    idx = np.arange(n)

    best_loss = float("inf")
    best_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        net.train()
        for i in range(0, n, batch_size):
            sl = idx[i:i + batch_size]
            xb = Xtr[sl]
            yb_cat = ytr_cat[sl]
            yb_glb = ytr_glb[sl]

            outs = net(xb)
            O = torch.cat(outs, dim=1)  # (B,6)
            pred_cat = O[:, :5]
            pred_glb = O[:, 5:6]

            loss_cat = huber_loss(pred_cat, yb_cat).mean()  # loss kategori
            loss_glb = huber_loss(pred_glb, yb_glb).mean() # loss global→label
            mean_cat = pred_cat.mean(dim=1, keepdim=True) # mean(5 kategori)
            loss_cons = huber_loss(pred_glb, mean_cat).mean() # loss konsistensi

            loss = loss_cat + alpha_glob * loss_glb + beta_cons * loss_cons # total loss

            opt.zero_grad()
            # menghitung setiap loss secara manual agar kompatibel dengan PyTorch versi lama
            loss.backward()
            opt.step()

        # valid
        net.eval()
        with torch.no_grad():
            Outs = net(Xva)
            Ova = torch.cat(Outs, dim=1)
            pred_glb_va = Ova[:, 5:6]
            # early-stop berdasar MSE global ke label
            val_loss = nn.functional.mse_loss(pred_glb_va, yva_glb).item()

        if val_loss + 1e-9 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # restore
    if best_state is not None:
        net.load_state_dict(best_state)

    # pred final (train+valid), untuk kalibrasi di luar
    net.eval()
    with torch.no_grad():
        preds_tr = torch.cat(net(Xtr), dim=1).cpu().numpy()  # (n,6)
        preds_va = torch.cat(net(Xva), dim=1).cpu().numpy()

    pack = {
        "state_dict": net.state_dict(),
        "best_val_loss": best_loss,
    }
    return preds_tr, preds_va, pack


# ------------------------------
# Kalibrasi per-head
# ------------------------------
def fit_isotonic(y_pred: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    # y_pred, y_true shape (N,)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_pred, y_true)
    return ir

def apply_isotonic(ir: IsotonicRegression, y_pred: np.ndarray) -> np.ndarray:
    return ir.transform(y_pred)


# ------------------------------
# Pipeline Train
# ------------------------------
def run_train(args):
    os.makedirs(args.artifact_dir, exist_ok=True)

    # 1) load data & schema
    df = pd.read_csv(args.data)
    orientation_map, category_map, CATEGORIES = load_schema(args.schema)

    # 2) siapkan fitur yang tersedia
    feat_all = [f for f in orientation_map.keys() if f in df.columns]
    if len(feat_all) == 0:
        raise ValueError("Tidak ada fitur dari schema yang ditemukan di data.")

    # 3) simpan schema untuk audit
    with open(os.path.join(args.artifact_dir, "schema_used.json"), "w") as f:
        json.dump({
            "categories": CATEGORIES,
            "orientation_map": orientation_map
        }, f, indent=2)

    # 4) fit normalizer orientasi
    norm_params = fit_normalizer(df, label_col=args.label_col, orientation_map=orientation_map)
    with open(os.path.join(args.artifact_dir, "normalizer.json"), "w") as f:
        json.dump(norm_params, f, indent=2)

    # 5) apply normalizer
    df_norm = apply_normalizer(df, norm_params)
    X_all = df_norm[feat_all].values.astype(np.float32)
    y_all = df[args.label_col].astype(int).values
    Y_glob = (y_all / 2.0).astype(np.float32).reshape(-1, 1)

    # 6) teacher deterministik per-kategori
    cat_det = aggregate_category_scores(df_norm, CATEGORIES, agg=args.cat_agg)
    # Pastikan urutan kategori tetap konsisten:
    CAT_ORDER = ["VisualClarity", "StructuralSimplicity", "DocumentationSupport",
                 "NamingTransparency", "CognitiveLoad"]
    for c in CAT_ORDER:
        if c not in cat_det.columns:
            cat_det[c] = 0.0
    Y_cat_det = cat_det[CAT_ORDER].values.astype(np.float32)

    # 7) KFold training
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    preds_oof = np.zeros((len(df), 6), dtype=np.float32)
    packs = []
    calibs = []  # list of per-fold isotonic list (6 head)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), 1):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr_cat, y_va_cat = Y_cat_det[tr_idx], Y_cat_det[va_idx]
        y_tr_glb, y_va_glb = Y_glob[tr_idx], Y_glob[va_idx]

        preds_tr, preds_va, pack = train_fold(
            X_tr, y_tr_cat, y_tr_glb,
            X_va, y_va_cat, y_va_glb,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            seed=42 + fold, alpha_glob=args.alpha_glob, beta_cons=args.beta_cons
        )

        # Fit isotonic per-head:
        # Head 0..4: target kategori teacher (kolom per-head)
        # Head 5   : target global (Y_glob)
        cal_fold = []
        for h in range(6):
            if h < 5:
                ytrue_tr = y_tr_cat[:, h]
                ypred_tr = preds_tr[:, h]
            else:
                ytrue_tr = y_tr_glb[:, 0]
                ypred_tr = preds_tr[:, h]

            try:
                ir = fit_isotonic(ypred_tr, ytrue_tr)
            except Exception:
                ir = None
            cal_fold.append(ir)

            # simpan OOF valid
            preds_oof[va_idx, h] = preds_va[:, h]

        packs.append(pack)
        calibs.append(cal_fold)

        print(f"[Fold {fold}] Best val MSE(Global): {pack['best_val_loss']:.6f}")

    # 8) Simpan artefak
    torch.save(packs, os.path.join(args.artifact_dir, "model_packs.pt"))
    # simpan kalibrator per-fold
    import pickle
    with open(os.path.join(args.artifact_dir, "calibrators.pkl"), "wb") as f:
        pickle.dump({"folds": calibs}, f)

    # 9) Simpan OOF + teacher deterministik (untuk sanity check orientasi)
    out_oof = pd.DataFrame({
        "idx": np.arange(len(df)),
        "Global_raw": preds_oof[:, 5],
    })
    for i, name in enumerate(["VisualClarity","StructuralSimplicity","DocumentationSupport",
                              "NamingTransparency","CognitiveLoad"]):
        out_oof[f"{name}_raw"] = preds_oof[:, i]
        out_oof[f"{name}_teacher"] = Y_cat_det[:, i]
    out_oof["Global_teacher_meanOfCat"] = Y_cat_det.mean(axis=1)
    out_path = os.path.join(args.artifact_dir, "oof_preds.csv")
    out_oof.to_csv(out_path, index=False)
    print(f"[Saved] OOF & teacher: {out_path}")


# ------------------------------
# Pipeline Infer
# ------------------------------
def run_infer(args):
    # Load artefak
    with open(os.path.join(args.artifact_dir, "schema_used.json"), "r") as f:
        sch = json.load(f)
    orientation_map = sch["orientation_map"]
    CATEGORIES = sch["categories"]

    with open(os.path.join(args.artifact_dir, "normalizer.json"), "r") as f:
        norm_params = json.load(f)

    packs = torch.load(os.path.join(args.artifact_dir, "model_packs.pt"), map_location="cpu") # type: ignore
    import pickle
    with open(os.path.join(args.artifact_dir, "calibrators.pkl"), "rb") as f:
        calib = pickle.load(f)["folds"]

    # Data
    df = pd.read_csv(args.data)
    feat_all = [f for f in orientation_map.keys() if f in df.columns]
    df_norm = apply_normalizer(df, norm_params)
    X = df_norm[feat_all].values.astype(np.float32)

    # Ensemble rata-rata antar fold
    preds_all = []
    for fold, pack in enumerate(packs, 1):
        in_dim = X.shape[1]
        net = MLPOriented(in_dim)
        net.load_state_dict(pack["state_dict"])
        net.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            P = torch.cat(net(Xt), dim=1).numpy()  # (N,6)
        # aplikasi kalibrator per-head (per-fold)
        cal_fold = calib[fold - 1]
        P_cal = P.copy()
        for h in range(6):
            ir = cal_fold[h]
            if ir is not None:
                P_cal[:, h] = apply_isotonic(ir, P[:, h])
        preds_all.append(P_cal)

    P_ens = np.mean(preds_all, axis=0)  # (N,6)

    # Tambahkan teacher deterministik (sanity-check orientasi) jika diminta
    cat_det = aggregate_category_scores(df_norm, CATEGORIES, agg=args.cat_agg)
    CAT_ORDER = ["VisualClarity", "StructuralSimplicity", "DocumentationSupport",
                 "NamingTransparency", "CognitiveLoad"]
    for c in CAT_ORDER:
        if c not in cat_det.columns: cat_det[c] = 0.0
    Y_cat_det = cat_det[CAT_ORDER].values.astype(np.float32)

    cols_data = {
        "GlobalScore": P_ens[:, 5],
        "VisualClarity": P_ens[:, 0],
        "StructuralSimplicity": P_ens[:, 1],
        "DocumentationSupport": P_ens[:, 2],
        "NamingTransparency": P_ens[:, 3],
        "CognitiveLoad": P_ens[:, 4],
    }
    out = pd.DataFrame(cols_data)

    # Prioritas 1: ambil id dari nmFile (format path/.../<id>.java)
    if "nmFile" in df.columns:
        out.insert(0, "id", df["nmFile"].apply(_extract_id_from_nmfile))
    else:
        # fallback: pakai id/idx/snippet_id bila nmFile tidak tersedia
        for c in ["id", "idx", "snippet_id"]:
            if c in df.columns:
                out.insert(0, "id", df[c].astype(str))
                break

    # optional: tulis teacher deterministik untuk evaluasi orientasi
    if args.dump_teacher:
        out["Teacher_VisualClarity"] = Y_cat_det[:, 0]
        out["Teacher_StructuralSimplicity"] = Y_cat_det[:, 1]
        out["Teacher_DocumentationSupport"] = Y_cat_det[:, 2]
        out["Teacher_NamingTransparency"] = Y_cat_det[:, 3]
        out["Teacher_CognitiveLoad"] = Y_cat_det[:, 4]

    out.to_csv(args.out, index=False)
    print(f"[Saved] Inference scores → {args.out}")

# --- util id dari nmFile ---
def _extract_id_from_nmfile(path_val: str) -> str:
    """
    Mengambil <id> dari kolom nmFile yang berisi path:
    /.../chunk_0095/<id>.java  ->  <id>
    Juga aman untuk path Windows dan ekstensi ganda.
    """
    try:
        s = str(path_val or "").strip()
        base = os.path.basename(s)           # <id>.java
        name, ext = os.path.splitext(base)   # (<id>, .java)
        # jika ada ekstensi ganda, kupas sampai habis
        while ext:
            base = name
            name, ext = os.path.splitext(base)
        return name if name else base
    except Exception:
        return str(path_val)
    
# ------------------------------
# Main / CLI
# ------------------------------
def main():
    p = argparse.ArgumentParser(description="KERTA DNN Sigmoid (Oriented Categories)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # TRAIN
    pt = sub.add_parser("train")
    pt.add_argument("--data", required=True, help="CSV data berisi fitur & label")
    pt.add_argument("--schema", required=True, help="CSV skema: feature,category,orientation")
    pt.add_argument("--label-col", default="target", help="Nama kolom label (0,1,2)")
    pt.add_argument("--artifact-dir", required=True, help="Folder simpan artefak")
    pt.add_argument("--n-folds", type=int, default=5)
    pt.add_argument("--epochs", type=int, default=200)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--lr", type=float, default=2e-3)
    pt.add_argument("--weight-decay", type=float, default=1e-4)
    pt.add_argument("--patience", type=int, default=25)
    pt.add_argument("--cat-agg", choices=["mean","median","harmonic"], default="mean",
                    help="Agregasi untuk teacher deterministik kategori")
    pt.add_argument("--alpha-glob", type=float, default=1.0,
                    help="Bobot loss global→label")
    pt.add_argument("--beta-cons", type=float, default=0.5,
                    help="Bobot regularisasi konsistensi global~mean(kategori)")

    # INFER
    pi = sub.add_parser("infer")
    pi.add_argument("--data", required=True, help="CSV data berisi fitur")
    pi.add_argument("--artifact-dir", required=True, help="Folder artefak hasil train")
    pi.add_argument("--out", required=True, help="CSV keluaran skor")
    pi.add_argument("--cat-agg", choices=["mean","median","harmonic"], default="mean",
                    help="(Opsional) hanya untuk menuliskan teacher deterministik")
    pi.add_argument("--dump-teacher", action="store_true",
                    help="Tulis skor teacher deterministik per-kategori (sanity-check orientasi)")

    args = p.parse_args()
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "infer":
        run_infer(args)
    else:
        raise ValueError("Unknown cmd")

if __name__ == "__main__":
    main()
