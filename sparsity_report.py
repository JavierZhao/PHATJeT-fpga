"""Layer-resolved kernel sparsity report for ProxL1 checkpoints.

For each kernel: exact float zeros and exact QUANTIZED zeros (the constants
da4ml actually receives). The two can differ in both directions — floats
below half an LSB quantize to zero without being zero, and (in principle)
rounding could resurrect a zero. The quantized column is the one hardware
cares about (Codex round-8 audit).

Usage: PJ_* env must match the checkpoint (as for extract_hw.py), then
    python sparsity_report.py <ckpt.keras> [<ckpt2.keras> ...]
"""
import os
import sys

import numpy as np
import keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_qat import build_qat_model  # noqa: E402

QKV_NAMES = ("wq", "wk", "wv")


def report(ckpt):
    model = build_qat_model()
    model.load_weights(ckpt)
    print(f"\n=== {ckpt}")
    rows = []
    for layer in model.layers:
        k = getattr(layer, "kernel", None)
        if k is None:
            continue
        w = keras.ops.convert_to_numpy(k)
        qk = getattr(layer, "qkernel", None)
        qw = keras.ops.convert_to_numpy(qk) if qk is not None else w
        rows.append((layer.name, w.size, float((w == 0).mean()), float((qw == 0).mean())))
    tot = sum(r[1] for r in rows)
    fz = sum(r[1] * r[2] for r in rows)
    qz = sum(r[1] * r[3] for r in rows)
    print(f"{'layer':32s} {'params':>8s} {'float0':>8s} {'quant0':>8s}")
    for name, n, f0, q0 in rows:
        tag = "  <-- QKV" if any(s in name for s in QKV_NAMES) else ""
        print(f"{name:32s} {n:8d} {f0:8.3f} {q0:8.3f}{tag}")
    print(f"{'TOTAL':32s} {tot:8d} {fz / tot:8.3f} {qz / tot:8.3f}")
    qkv_n = sum(n for name, n, _, _ in rows if any(s in name for s in QKV_NAMES))
    qkv_qz = sum(n * q0 for name, n, _, q0 in rows if any(s in name for s in QKV_NAMES))
    print(f"QKV-only quantized zeros: {qkv_qz / max(qkv_n, 1):.3f} over {qkv_n} params")


if __name__ == "__main__":
    for c in sys.argv[1:]:
        report(c)
