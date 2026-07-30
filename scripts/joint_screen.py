#!/usr/bin/env python3
"""Joint (non-OFAT) screen over rp's design space, with fair-shake hyperparameters.

Addresses the four gaps of the earlier single-factor sweeps:
  1. interactions: full cross of basis x weak-kernel x strong-kernel x shared reg, plus
     mixed bases (raw weak + SAE strong) and a strong-side truncation branch;
  2. fair shake: every combo is scored at ITS OWN regularization (trace-scaled grid),
     not at rp's reg=0.1;
  3. finer grids: 7 bandwidths, 7 regs, 7 truncation ranks;
  4. proxy blindness: tickets to a REAL downstream run are issued by a preregistered rule
     that includes a diversity quota (best combo with top-50 overlap vs rp < 0.85), because
     mlpstep1 proved the proxy cannot see downstream winners.

Outputs: a report json, a printed table of the frontier, and custom_scores npz (cs1..cs4 +
weak_preds for the pipeline's alignment guard) for the downstream stage.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from compute_pgr import auroc
import torch


def rankv(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def build_kernel(X: torch.Tensor, kind: str, c: float | None, H: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    if kind == "lin":
        return X @ X.T / n
    sq = (X * X).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2 * (X @ X.T)).clamp_min_(0)
    iu = torch.triu_indices(n, n, offset=1)
    med = d2[iu[0], iu[1]].sqrt().median()
    K = torch.exp(-d2 / (2 * (c * med) ** 2))
    return H @ K @ H  # double-center: keep the constant-annihilation property


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="acts_r1_seed42.npz")
    ap.add_argument("--bulk", default="bulk_acts_r1.npz")
    ap.add_argument("--out-scores", default="custom_scores_r1.npz")
    ap.add_argument("--out-report", default="joint_screen_report.json")
    ap.add_argument("--no-sae", action="store_true", help="raw basis only (smoke)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    lab = np.load(args.labeled)
    Xw_raw = lab["weak"].astype(np.float32)
    Xs_raw = lab["strong"].astype(np.float32)
    yhat = lab["weak_preds"].astype(np.float64)
    wrong = lab["weak_preds"].astype(int) != lab["gt"].astype(int)
    n = len(yhat)
    yc_np = yhat - yhat.mean()
    yc = torch.tensor(yc_np, dtype=torch.float32, device=dev)
    H = (torch.eye(n) - torch.ones(n, n) / n).to(dev)

    def center(M):
        return M - M.mean(0, keepdims=True)

    bases: dict[str, tuple[np.ndarray, np.ndarray]] = {"raw": (center(Xw_raw), center(Xs_raw))}
    if not args.no_sae:
        import sys
        sys.path.insert(0, "scripts")
        from sae_alignment_screen import train_sae, encode
        bulk = np.load(args.bulk)
        for tag, l1 in (("saeA", 1e-3), ("saeB", 3e-3)):
            sw, muw, sdw, _ = train_sae(bulk["weak"].astype(np.float32), 8 * 896, l1, dev, seed=1)
            ss, mus, sds, _ = train_sae(bulk["strong"].astype(np.float32), 4 * 3584, l1, dev, seed=2)
            Zw = center(encode(sw, Xw_raw, muw, sdw, dev).astype(np.float32))
            Zs = center(encode(ss, Xs_raw, mus, sds, dev).astype(np.float32))
            bases[tag] = (Zw, Zs)
            bases[f"mix_{tag}"] = (center(Xw_raw), Zs)  # interaction cell: raw weak + SAE strong
            print(f"[basis] {tag} ready", flush=True)

    KIND = [("lin", None)] + [("rbf", c) for c in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)]
    REGS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    TOPK = (16, 32, 64, 128, 256, 512, 1024)

    # rp reference (raw, lin/lin, reg=.1)
    def smooth_np(K, t, reg):
        r = reg * float(np.trace(K)) / n + 1e-12
        return K @ np.linalg.solve(K + r * np.eye(n), t)
    Kw0 = (bases["raw"][0] @ bases["raw"][0].T / n).astype(np.float64)
    Ks0 = (bases["raw"][1] @ bases["raw"][1].T / n).astype(np.float64)
    rp_ref = np.abs(smooth_np(Ks0, yc_np - smooth_np(Kw0, yc_np, 0.1), 0.1))
    top_rp = set(np.argsort(-rp_ref)[: n // 2].tolist())
    rp_rank = rankv(rp_ref)

    def metrics(score: np.ndarray):
        au = auroc(score, wrong.astype(int))
        ov = len(set(np.argsort(-score)[: n // 2].tolist()) & top_rp) / (n // 2)
        return au, abs(au - 0.5), ov

    au0, sep0, _ = metrics(rp_ref)
    print(f"rp reference: errAUROC {au0:.3f} sep {sep0:.3f}", flush=True)

    rows, vectors = [], {}
    for bname, (Bw, Bs) in bases.items():
        tw = torch.tensor(Bw, device=dev); ts = torch.tensor(Bs, device=dev)
        eigs = {}
        for side, T in (("w", tw), ("s", ts)):
            for kind, c in KIND:
                K = build_kernel(T, kind, c, H)
                d, U = torch.linalg.eigh(K)
                eigs[(side, kind, c)] = (d.clamp_min(0), U)
        print(f"[eig] {bname}: {len(eigs)} kernels decomposed", flush=True)

        def smooth_eig(key, t, reg):
            d, U = eigs[key]
            r = reg * d.mean() + 1e-12
            return U @ ((d / (d + r)) * (U.T @ t))

        for wk in KIND:
            for sk in KIND:
                for reg in REGS:
                    a = yc - smooth_eig(("w", *wk), yc, reg)
                    v = smooth_eig(("s", *sk), a, reg)
                    sc = v.abs().cpu().numpy().astype(np.float64)
                    au, sep, ov = metrics(sc)
                    rows.append({"basis": bname, "wk": f"{wk[0]}{wk[1] or ''}",
                                 "sk": f"{sk[0]}{sk[1] or ''}", "reg": reg, "branch": "ridge",
                                 "auroc": au, "sep": sep, "overlap": ov})
                    vectors[len(rows) - 1] = sc
        # truncation branch: weak side lin (own reg), strong side hard top-k
        _, Us = eigs[("s", "lin", None)]
        for reg in REGS:
            a = yc - smooth_eig(("w", "lin", None), yc, reg)
            for k in TOPK:
                Uk = Us[:, -k:]
                sc = (Uk @ (Uk.T @ a)).abs().cpu().numpy().astype(np.float64)
                au, sep, ov = metrics(sc)
                rows.append({"basis": bname, "wk": f"lin(r{reg})", "sk": f"top{k}",
                             "reg": reg, "branch": "topk", "auroc": au, "sep": sep, "overlap": ov})
                vectors[len(rows) - 1] = sc

    order = sorted(range(len(rows)), key=lambda i: -rows[i]["sep"])
    print("\n== frontier (top 12 by separation) ==")
    for i in order[:12]:
        r = rows[i]
        print(f"  {r['basis']:9s} wk={r['wk']:8s} sk={r['sk']:8s} reg={r['reg']:<6g} "
              f"sep={r['sep']:.3f} auroc={r['auroc']:.3f} overlap={r['overlap']:.3f}")

    # preregistered ticket rule
    def best(pred):
        cand = [i for i in order if pred(rows[i])]
        return cand[0] if cand else None
    t1 = order[0]
    t2 = best(lambda r: r["overlap"] < 0.85)
    t3 = best(lambda r: r["basis"] != "raw")
    t4 = best(lambda r: r["basis"] == "raw" and (r["wk"].startswith("rbf") or r["sk"].startswith("rbf")))
    tickets = []
    for t in (t1, t2, t3, t4):
        if t is not None and t not in tickets:
            tickets.append(t)
    for i in order:
        if len(tickets) >= 4:
            break
        if i not in tickets:
            tickets.append(i)

    out = {"weak_preds": lab["weak_preds"]}
    tinfo = []
    for slot, i in enumerate(tickets, 1):
        out[f"cs{slot}"] = vectors[i]
        tinfo.append({"slot": f"cs{slot}", **rows[i]})
        print(f"TICKET cs{slot}: {rows[i]}")
    np.savez_compressed(args.out_scores, **out)
    with open(args.out_report, "w") as f:
        json.dump({"rp_sep": sep0, "n_combos": len(rows), "tickets": tinfo,
                   "frontier": [rows[i] for i in order[:40]]}, f, indent=1)
    print(f"DONE combos={len(rows)}")


if __name__ == "__main__":
    main()
