"""
Build the kt-sorted 150-constituent training set for PHAT-JeT QAT.

WHY THIS IS NOT A ONE-LINER
---------------------------
The processed store has `x_train_robust_150const_ptetaphi_sorted_kt.npy` but
there is NO sorted validation file -- only `x_val_robust_150const_ptetaphi.npy`
(unsorted). Training on kt-sorted inputs and validating on unsorted ones would
be a silent train/serve skew, and it matters more here than for a permutation-
invariant model: PHAT-JeT's patches are *contiguous slices* of the sequence, so
the ordering determines which particles attend to each other.

So we must sort the val set ourselves. To avoid inventing an ordering that
disagrees with the one used for train, we first REVERSE-ENGINEER the train
sort key: sorting is a within-jet permutation, so for each candidate key we
check whether applying it to the unsorted train file reproduces the provided
sorted file exactly. Whichever key reproduces it is then applied to val.

If no candidate reproduces the train file, we fall back to sorting BOTH train
and val ourselves with one key -- self-consistent by construction, at the cost
of not matching the paper's exact ordering. Either way the outputs are
consistent; the log says which path was taken.

Outputs <out>/jets_150x3_kt.npz with x_train, y_train, x_val, y_val.

Usage:  python prep_data.py --src <processed dir> --out <dir> [--sample 4000]
"""

import argparse
import os

import numpy as np

FEAT = ("pt", "deta", "dphi")


def candidate_keys():
    """Descending-sort keys to test, as functions of x[...,3] -> score."""

    def dr(x):
        return np.sqrt(x[..., 1] ** 2 + x[..., 2] ** 2)

    return {
        "pt*dR": lambda x: x[..., 0] * dr(x),
        "pt": lambda x: x[..., 0],
        "dR": lambda x: dr(x),
        "exp(pt)*dR": lambda x: np.exp(x[..., 0]) * dr(x),
        "pt*dR^2": lambda x: x[..., 0] * dr(x) ** 2,
        "-dR": lambda x: -dr(x),
    }


def apply_sort(x, key_fn):
    """Sort constituents within each jet by key, descending. Stable."""
    k = key_fn(x)
    order = np.argsort(-k, axis=1, kind="stable")
    return np.take_along_axis(x, order[..., None], axis=1)


def identify_key(x_unsorted, x_sorted, sample):
    """Return (name, fn) of the candidate that reproduces x_sorted, else None."""
    xs_u, xs_s = x_unsorted[:sample], x_sorted[:sample]
    print(f"probing sort key on {sample:,} jets")
    best = None
    for name, fn in candidate_keys().items():
        got = apply_sort(xs_u, fn)
        frac = float(np.isclose(got, xs_s, rtol=0, atol=1e-9).all(axis=(1, 2)).mean())
        print(f"  {name:<12} exact-jet match {frac:7.2%}")
        if best is None or frac > best[2]:
            best = (name, fn, frac)
    if best and best[2] >= 0.999:
        print(f"-> train file was sorted by {best[0]!r} ({best[2]:.2%})")
        return best[0], best[1]
    print(f"-> no candidate reproduces the train file (best {best[0]!r} at {best[2]:.2%})")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="processed/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=4000, help="jets used to probe the sort key")
    args = ap.parse_args()

    p = lambda f: os.path.join(args.src, f)
    stem = "robust_150const_ptetaphi"

    print("loading …")
    x_tr_sorted = np.load(p(f"x_train_{stem}_sorted_kt.npy"))
    x_tr_plain = np.load(p(f"x_train_{stem}.npy"))
    y_tr = np.load(p(f"y_train_{stem}.npy"))
    x_va_plain = np.load(p(f"x_val_{stem}.npy"))
    y_va = np.load(p(f"y_val_{stem}.npy"))
    print(f"  train {x_tr_sorted.shape}  val {x_va_plain.shape}")

    for nm, a in (("x_train_sorted", x_tr_sorted), ("x_val", x_va_plain)):
        print(f"  {nm}: {FEAT[0]} [{a[...,0].min():.3f},{a[...,0].max():.3f}] "
              f"{FEAT[1]} [{a[...,1].min():.3f},{a[...,1].max():.3f}] "
              f"{FEAT[2]} [{a[...,2].min():.3f},{a[...,2].max():.3f}]")

    found = identify_key(x_tr_plain, x_tr_sorted, args.sample)

    if found is not None:
        key_name, key_fn = found
        x_train = x_tr_sorted                       # use the provided file verbatim
        x_val = apply_sort(x_va_plain, key_fn)      # same key applied to val
        provenance = f"train=provided sorted_kt; val=sorted here by {key_name!r}"
    else:
        key_name, key_fn = "pt*dR", candidate_keys()["pt*dR"]
        x_train = apply_sort(x_tr_plain, key_fn)
        x_val = apply_sort(x_va_plain, key_fn)
        provenance = (f"provided sorted_kt NOT reproducible; both splits sorted here "
                      f"by {key_name!r} (self-consistent, may differ from the paper)")
    print(f"provenance: {provenance}")

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "jets_150x3_kt.npz")
    np.savez(
        dst,
        x_train=x_train.astype(np.float32),
        y_train=y_tr.astype(np.float32),
        x_val=x_val.astype(np.float32),
        y_val=y_va.astype(np.float32),
        provenance=np.array(provenance),
    )
    print(f"wrote {dst}  ({os.path.getsize(dst)/1e9:.2f} GB)")
    print(f"  x_train {x_train.shape}  x_val {x_val.shape}")


if __name__ == "__main__":
    main()
