"""D2 offline supervision: per-column MLP fit to physics tendencies from E9.

One arm of the D8 bridging experiment. Trains a per-column MLP to reproduce the
DCMIP-2016 test 1 ground-truth tendencies loaded by `dcmip16_data.TendencyDataset`.
The other arm (F3 trajectory-matching via `forward()` + `harness.autograd`) reuses
the same NN architecture — see `harness.py`.

Usage:
    python train_offline.py \\
        --data '/path/to/movies/dcmip16-t1-ne30-hifreq-*.nc' \\
        --heads fm_x,fm_y,fm_z,fvtheta \\
        --epochs 5 --batch-cols 4096

Heads: subset of dcmip16_data.ALL_HEADS. Default = BINDABLE_HEADS
(fm_{x,y,z}, fvtheta). Pass fq_qv,fq_qc,fq_qr as well to also train tracer heads
(dataset-only until B9 binds them into pyhommexx forcing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dcmip16_data import ALL_HEADS, BINDABLE_HEADS, INPUT_STATE_VARS, TendencyDataset


class PerColumnMLP(torch.nn.Module):
    """Column-local MLP: (ncol, nlev * n_state_vars) -> (ncol, nlev * n_heads).

    Column-local is the D5 architectural constraint. If this ever becomes a CNN
    or transformer with a horizontal receptive field, DSS reconciliation stops
    being a no-op (see plan §2).
    """

    def __init__(self, nlev: int, n_state: int, n_heads: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        in_dim  = nlev * n_state
        out_dim = nlev * n_heads
        layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, hidden), torch.nn.GELU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.GELU()]
        layers += [torch.nn.Linear(hidden, out_dim)]
        self.net = torch.nn.Sequential(*layers)
        self.nlev = nlev
        self.n_heads = n_heads

    def forward(self, cols_flat: torch.Tensor) -> torch.Tensor:  # (ncol, in_dim) -> (ncol, nlev, n_heads)
        out = self.net(cols_flat)
        return out.view(-1, self.nlev, self.n_heads)


def _stack_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """(ncol, nlev) per var -> (ncol, nlev * n_state) with a fixed field order."""
    # w and phi are interface-level (nlev+1). Truncate to midpoints for the input encoding.
    # This is a defensible starting choice; per-level interp is D-series work.
    cols = []
    for name in INPUT_STATE_VARS:
        x = state[name]
        if x.shape[-1] != state["u"].shape[-1]:
            x = x[..., : state["u"].shape[-1]]
        cols.append(x)
    return torch.cat(cols, dim=-1)


def _stack_targets(tendency: dict[str, torch.Tensor], heads: list[str]) -> torch.Tensor:
    """(ncol, nlev) per head -> (ncol, nlev, n_heads)."""
    return torch.stack([tendency[h] for h in heads], dim=-1)


def train(args: argparse.Namespace) -> None:
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    for h in heads:
        if h not in ALL_HEADS:
            raise SystemExit(f"unknown head {h!r}; valid: {ALL_HEADS}")
    unbindable = [h for h in heads if h not in BINDABLE_HEADS]
    if unbindable:
        print(f"[train_offline] note: heads {unbindable} have no pyhommexx binding yet "
              "(B9 blocked). Offline D2 training runs; F3 trajectory arm can't consume them "
              "until bindings land.")

    ds = TendencyDataset(args.data, head_names=heads, stride=args.stride)
    print(f"[train_offline] dataset: {len(ds)} snapshots  heads={heads}")

    sample0 = ds[0]
    nlev = sample0.state["u"].shape[-1]
    model = PerColumnMLP(nlev=nlev, n_state=len(INPUT_STATE_VARS), n_heads=len(heads),
                         hidden=args.hidden, depth=args.depth)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DataLoader yields lists of Sample; collate ourselves so column slicing stays explicit.
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda xs: xs[0])

    for epoch in range(args.epochs):
        running = 0.0
        n_batches = 0
        for sample in loader:
            x = _stack_state(sample.state)              # (ncol, nlev*n_state)
            y = _stack_targets(sample.tendency, heads)  # (ncol, nlev, n_heads)
            ncol = x.shape[0]

            # column-level minibatching (memory + regularization)
            perm = torch.randperm(ncol)
            for i in range(0, ncol, args.batch_cols):
                ix = perm[i : i + args.batch_cols]
                pred = model(x[ix])
                loss = torch.nn.functional.mse_loss(pred, y[ix])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += loss.item()
                n_batches += 1

        print(f"epoch {epoch:3d}  mean MSE = {running / max(n_batches, 1):.4e}")

    torch.save({"state_dict": model.state_dict(), "heads": heads, "nlev": nlev}, args.out)
    print(f"[train_offline] wrote {args.out}")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="glob pattern for E9 NetCDFs")
    p.add_argument("--heads", default=",".join(BINDABLE_HEADS),
                   help=f"comma-separated subset of {ALL_HEADS}")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-cols", type=int, default=4096)
    p.add_argument("--stride", type=int, default=1, help="subsample every Nth time snapshot")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="d2_offline.pt")
    return p.parse_args()


if __name__ == "__main__":
    train(_cli())
