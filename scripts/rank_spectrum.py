#!/usr/bin/env python3
"""Cumulative spectral energy of a weight update.

For a weight-update matrix dW, let s_1 >= s_2 >= ... >= s_m be its singular
values and define

    E(k) = sum_{i <= k} s_i^2 / sum_i s_i^2 .

E(k) is the fraction of squared Frobenius energy captured by the best rank-k
approximation. The effective rank at threshold eta is

    k_eta = min { k : E(k) >= eta } .

Eckart-Young-Mirsky gives the exact relative Frobenius error of that
approximation:

    ||dW - dW_k||_F / ||dW||_F = sqrt(1 - E(k)) .

For a PEFT LoRA adapter the update is dW = (alpha / r) * B @ A. E(k) does not
depend on the global scalar alpha / r, so it is invariant to the adapter
scaling; the scaling only matters for absolute magnitudes.

Modes of operation
------------------
--selftest
    Build a synthetic dW with a known rank and singular values, then verify
    that the spectrum, k_eta, the Eckart-Young bound, the LoRA scaling, and
    the safetensors reader all recover what was put in. Exits non-zero on any
    failure.
--adapter PATH --base PATH
    Analyze a PEFT adapter against its base checkpoint. Handles LoRA pairs
    (lora_A / lora_B) and modules_to_save full-weight tensors (e.g. an
    unfrozen vision tower), reading both files with a small numpy-only
    safetensors parser.
--npz PATH
    Analyze every 2-D array in a .npz archive (one matrix per entry).

Examples
--------
    python scripts/rank_spectrum.py --selftest
    python scripts/rank_spectrum.py --adapter adapter_model.safetensors \
        --base model.safetensors --lora-alpha 8 --lora-r 32
    python scripts/rank_spectrum.py --npz deltas.npz --json
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

LORA_A = ".lora_A.weight"
LORA_B = ".lora_B.weight"
MODULES_TO_SAVE = ".modules_to_save.default"
ADAPTER_PREFIX = "base_model.model."


# ---------------------------------------------------------------------------
# Core spectrum math
# ---------------------------------------------------------------------------


def energy_curve(s: np.ndarray) -> np.ndarray:
    """Cumulative spectral energy E(k) for k = 1..len(s)."""
    s = np.asarray(s, dtype=np.float64).ravel()
    total = float(np.sum(s * s))
    if total == 0.0:
        return np.zeros_like(s)
    return np.cumsum(s * s) / total


def k_for_energy(energy: np.ndarray, eta: float = 0.99) -> int:
    """Smallest k with E(k) >= eta; 0 for a zero matrix."""
    energy = np.asarray(energy, dtype=np.float64)
    if energy.size == 0 or energy[-1] <= 0.0:
        return 0
    if eta > 1.0:
        raise ValueError(f"eta must be <= 1, got {eta}")
    if eta <= 0.0:
        return 1
    return int(np.searchsorted(energy, eta, side="left")) + 1


def eckart_young_error(energy: np.ndarray, k: int) -> float:
    """Best rank-k relative Frobenius error sqrt(1 - E(k))."""
    energy = np.asarray(energy, dtype=np.float64)
    if energy.size == 0 or energy[-1] <= 0.0:
        return 0.0
    e = 1.0 if k >= energy.size else float(energy[k - 1])
    return float(np.sqrt(max(0.0, 1.0 - e)))


def spectrum(dW: np.ndarray, eta: float = 0.99, probes: tuple[int, ...] = (8, 16, 24)) -> dict:
    """Singular values, energy curve, k_eta, and error bounds for a matrix."""
    dW = np.asarray(dW, dtype=np.float64)
    if dW.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {dW.shape}")
    s = np.linalg.svd(dW, compute_uv=False)
    energy = energy_curve(s)
    out = {
        "shape": list(dW.shape),
        "rank": int(np.sum(s > s[0] * 1e-12)) if s.size and s[0] > 0 else 0,
        "fro": float(np.sqrt(np.sum(s * s))),
        "k": int(k_for_energy(energy, eta)),
        "eta": float(eta),
        "energy": {str(k): (1.0 if k >= energy.size else float(energy[k - 1])) for k in probes},
        "error_bound": {str(k): eckart_young_error(energy, k) for k in probes},
    }
    return out


def lora_delta_w(a: np.ndarray, b: np.ndarray, alpha: float, r: int) -> np.ndarray:
    """PEFT LoRA update: dW = (alpha / r) * B @ A with A: [r, in], B: [out, r]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2-D")
    if a.shape[0] != b.shape[1]:
        raise ValueError(f"rank mismatch: A {a.shape} vs B {b.shape}")
    if r <= 0:
        raise ValueError("r must be positive")
    return (float(alpha) / float(r)) * (b @ a)


# ---------------------------------------------------------------------------
# Minimal numpy-only safetensors reader
# ---------------------------------------------------------------------------

_DTYPE_CODES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "?",
}


def _dtype_code(name: str) -> str:
    if name in _DTYPE_CODES:
        return _DTYPE_CODES[name]
    raise ValueError(f"unsupported safetensors dtype: {name}")


def read_safetensors_header(path: str | Path) -> tuple[dict, int]:
    """Return (header dict, byte offset where tensor data starts)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"not a safetensors file: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len > 100_000_000:
            raise ValueError(f"implausible safetensors header length {header_len}")
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header, 8 + header_len


def read_safetensors_tensor(
    path: str | Path, key: str, header: dict | None = None, data_start: int | None = None
) -> np.ndarray:
    """Read one tensor by key. BF16 is widened to float32 via a 16-bit left shift."""
    if header is None or data_start is None:
        header, data_start = read_safetensors_header(path)
    info = header[key]
    shape = tuple(int(d) for d in info["shape"])
    begin, end = (int(x) for x in info["data_offsets"])
    count = int(np.prod(shape)) if shape else 1
    if info["dtype"] == "BF16":
        bits = np.fromfile(path, dtype="<u2", count=count, offset=data_start + begin)
        if bits.size != count:
            raise ValueError(f"short read for {key}")
        values = (bits.astype(np.uint32) << 16).view(np.float32)
    else:
        values = np.fromfile(
            path, dtype=np.dtype(_dtype_code(info["dtype"])), count=count, offset=data_start + begin
        )
        if values.size != count:
            raise ValueError(f"short read for {key}")
    return values.reshape(shape)


def base_key_for(adapter_key: str) -> tuple[str, str] | None:
    """Map an adapter tensor key to (kind, base key), or None if unsupported."""
    key = adapter_key
    if key.startswith(ADAPTER_PREFIX):
        key = key[len(ADAPTER_PREFIX) :]
    if key.endswith(LORA_A) or key.endswith(LORA_B):
        stem = key[: -len(".lora_A.weight")] if key.endswith(LORA_A) else key[: -len(".lora_B.weight")]
        return "lora", stem + ".weight"
    if MODULES_TO_SAVE in key:
        return "full", key.replace(MODULES_TO_SAVE, "")
    return None


# ---------------------------------------------------------------------------
# Adapter / archive analysis
# ---------------------------------------------------------------------------


def analyze_adapter(
    adapter_path: Path,
    base_path: Path,
    lora_alpha: float | None,
    lora_r: int | None,
    eta: float,
    key_filter: str | None,
    max_modules: int | None,
) -> list[dict]:
    adapter_header, adapter_start = read_safetensors_header(adapter_path)
    base_header, base_start = read_safetensors_header(base_path)
    pattern = re.compile(key_filter) if key_filter else None

    a_keys = sorted(k for k in adapter_header if k.endswith(LORA_A))
    full_keys = sorted(k for k in adapter_header if MODULES_TO_SAVE in k)
    rows: list[dict] = []

    for a_key in a_keys:
        if pattern and not pattern.search(a_key):
            continue
        mapping = base_key_for(a_key)
        if mapping is None:
            continue
        _, base_key = mapping
        b_key = a_key[: -len(LORA_A)] + LORA_B
        if b_key not in adapter_header or base_key not in base_header:
            continue
        a = read_safetensors_tensor(adapter_path, a_key, adapter_header, adapter_start)
        b = read_safetensors_tensor(adapter_path, b_key, adapter_header, adapter_start)
        r = lora_r if lora_r is not None else int(a.shape[0])
        alpha = float(lora_alpha) if lora_alpha is not None else float(r)
        dW = lora_delta_w(a, b, alpha, r)
        info = spectrum(dW, eta=eta)
        info.update({"module": base_key, "kind": "lora", "r": r, "alpha": alpha})
        rows.append(info)

    for key in full_keys:
        if pattern and not pattern.search(key):
            continue
        mapping = base_key_for(key)
        if mapping is None:
            continue
        _, base_key = mapping
        if base_key not in base_header:
            continue
        w_ft = read_safetensors_tensor(adapter_path, key, adapter_header, adapter_start)
        w_base = read_safetensors_tensor(base_path, base_key, base_header, base_start)
        if w_ft.shape != w_base.shape:
            continue
        dW = w_ft.astype(np.float64) - w_base.astype(np.float64)
        info = spectrum(dW, eta=eta)
        info.update({"module": base_key, "kind": "full", "r": None, "alpha": None})
        rows.append(info)

    if not a_keys and not full_keys:
        raise SystemExit(f"no LoRA or modules_to_save tensors found in {adapter_path}")
    rows.sort(key=lambda r: r["k"])
    if max_modules is not None:
        rows = rows[:max_modules]
    return rows


def analyze_npz(path: Path, eta: float) -> list[dict]:
    data = np.load(path)
    rows = []
    for key in sorted(data.files):
        arr = data[key]
        if arr.ndim != 2:
            continue
        info = spectrum(arr, eta=eta)
        info.update({"module": key, "kind": "dense", "r": None, "alpha": None})
        rows.append(info)
    rows.sort(key=lambda r: r["k"])
    return rows


def _short_name(name: str, width: int = 62) -> str:
    """Keep the tail of long module names so the leaf path stays visible."""
    if len(name) <= width:
        return name
    return "..." + name[-(width - 3) :]


def print_rows(rows: list[dict], probe_keys: tuple[int, ...]) -> None:
    header = f"{'module':<62} {'shape':>12} {'k@eta':>6}"
    for k in probe_keys:
        header += f" {'E(' + str(k) + ')':>8}"
    header += f" {'err@max':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        shape = f"{r['shape'][0]}x{r['shape'][1]}"
        values = "".join(f" {r['energy'][str(k)]:>8.4f}" for k in probe_keys)
        err = r["error_bound"][str(max(probe_keys))]
        print(f"{_short_name(r['module']):<62} {shape:>12} {r['k']:>6}{values} {err:>8.4f}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Write a minimal safetensors file from {name: (dtype_name, array)}."""
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (dtype_name, arr) in tensors.items():
        blob = arr.tobytes()
        header[name] = {"dtype": dtype_name, "shape": list(arr.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw_header)))
        f.write(raw_header)
        f.write(b"".join(blobs))


def selftest() -> None:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    rng = np.random.default_rng(0)
    m, n, r_true = 96, 64, 8
    q_left = np.linalg.qr(rng.normal(size=(m, r_true)))[0]
    q_right = np.linalg.qr(rng.normal(size=(n, r_true)))[0]
    s_true = np.array([8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
    dW = (q_left * s_true) @ q_right.T

    info = spectrum(dW, eta=0.99, probes=(2, 4, 8, 12))
    s_hat = np.linalg.svd(dW, compute_uv=False)[:r_true]
    rel = float(np.max(np.abs(s_hat - s_true) / s_true))
    check("singular values of rank-8 synthetic dW", rel < 1e-10, f"max relative error {rel:.2e}")

    e = energy_curve(np.linalg.svd(dW, compute_uv=False))
    expected_k = k_for_energy(energy_curve(s_true), 0.99)
    check(
        "k@eta=0.99 on rank-8 dW",
        info["k"] == expected_k == 6,
        f"recovered k={info['k']}, expected from singular values={expected_k} "
        "(6 of 8 directions already carry 99% of the energy)",
    )
    check("E(8)=1 for a rank-8 matrix", abs(info["energy"]["8"] - 1.0) < 1e-12, f"E(8)={info['energy']['8']:.12f}")
    check("energy curve non-decreasing", bool(np.all(np.diff(e) >= -1e-15)))
    check("E(last)=1", abs(float(e[-1]) - 1.0) < 1e-12)

    for k in (1, 2, 4):
        dW_k = (q_left[:, :k] * s_true[:k]) @ q_right[:, :k].T
        measured = float(np.linalg.norm(dW - dW_k) / np.linalg.norm(dW))
        bound = eckart_young_error(energy_curve(s_true), k)
        ok = abs(measured - bound) < 1e-10
        check(
            f"Eckart-Young bound at k={k}",
            ok,
            f"measured {measured:.8f} vs sqrt(1-E({k})) {bound:.8f}",
        )

    # LoRA scaling and scale invariance of the spectrum.
    a = rng.normal(size=(32, 40))
    b = rng.normal(size=(50, 32))
    got = lora_delta_w(a, b, alpha=8, r=32)
    want = (8.0 / 32.0) * (b @ a)
    check("LoRA dW = (alpha/r) B A", np.allclose(got, want), f"max abs diff {np.max(np.abs(got - want)):.2e}")
    scaled = spectrum(10.0 * dW, eta=0.99, probes=(2, 4, 8, 12))
    check(
        "spectrum is invariant to the global scale alpha/r",
        scaled["k"] == info["k"] and np.allclose(list(scaled["energy"].values()), list(info["energy"].values())),
    )

    zero = spectrum(np.zeros((16, 16)), eta=0.99)
    check("zero matrix handled", zero["k"] == 0 and zero["error_bound"]["8"] == 0.0)

    # safetensors parser round-trip, including BF16 widening.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.safetensors"
        f32 = np.array([[1.0, 2.5], [-0.5, 3.0]], dtype=np.float32)
        bf16_values = np.array([1.0, 2.5, -0.5, 3.0, 0.25, -0.25], dtype=np.float32)
        bf16_bits = (bf16_values.view(np.uint32) >> 16).astype("<u2")
        _write_safetensors(
            path,
            {"test.f32": ("F32", f32), "test.bf16": ("BF16", bf16_bits.reshape(2, 3))},
        )
        header, start = read_safetensors_header(path)
        back_f32 = read_safetensors_tensor(path, "test.f32", header, start)
        check("safetensors f32 round-trip", np.array_equal(back_f32, f32))
        back_bf16 = read_safetensors_tensor(path, "test.bf16", header, start)
        check(
            "safetensors bf16 widening",
            np.array_equal(back_bf16.ravel(), bf16_values),
            str(back_bf16.ravel().tolist()),
        )

    print()
    if failures:
        print(f"[selftest] {len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
        raise SystemExit(1)
    print("[selftest] ALL CHECKS PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="run synthetic recovery checks and exit")
    parser.add_argument("--adapter", type=Path, help="PEFT adapter safetensors file")
    parser.add_argument("--base", type=Path, help="base checkpoint safetensors file")
    parser.add_argument("--npz", type=Path, help="npz archive of dense matrices")
    parser.add_argument("--lora-alpha", type=float, default=None, help="PEFT lora_alpha (default: r)")
    parser.add_argument("--lora-r", type=int, default=None, help="PEFT r (default: inferred from lora_A shape)")
    parser.add_argument("--eta", type=float, default=0.99, help="energy threshold for k_eta (default 0.99)")
    parser.add_argument("--key-filter", type=str, default=None, help="regex on adapter keys")
    parser.add_argument("--max-modules", type=int, default=None, help="print at most N modules")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    if not args.adapter and not args.npz:
        parser.error("choose a mode: --selftest, --adapter/--base, or --npz")

    if args.adapter:
        if not args.base:
            parser.error("--adapter requires --base")
        rows = analyze_adapter(
            args.adapter, args.base, args.lora_alpha, args.lora_r, args.eta, args.key_filter, args.max_modules
        )
    else:
        rows = analyze_npz(args.npz, args.eta)

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
        return 0

    probes = (8, 16, 24)
    print_rows(rows, probes)
    print()
    print(f"modules analyzed: {len(rows)}")
    print(f"eta: {args.eta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
