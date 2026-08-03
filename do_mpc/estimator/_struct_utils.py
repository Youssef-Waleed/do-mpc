"""Small CasADi-struct helpers used by the estimator."""

from __future__ import annotations

from typing import Any

import numpy as np


def dict_to_struct_column(
    val: dict[str, Any],
    struct,
    *,
    label: str,
) -> np.ndarray:
    """Order a name-keyed dict into the column layout of a CasADi struct."""
    keys = [k for k in struct.keys() if k != "default"]
    provided = set(val.keys())
    expected = set(keys)
    missing = expected - provided
    unknown = provided - expected
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if unknown:
            parts.append(f"unknown keys: {sorted(unknown)}")
        raise ValueError(
            f"{label} dict assignment requires exactly the struct entries "
            f"({sorted(expected)}); " + "; ".join(parts)
        )

    flat: list[float] = []
    for name in keys:
        slot = struct[name]
        size = slot.shape[0] * slot.shape[1] if hasattr(slot, "shape") else 1
        entry = np.atleast_1d(np.asarray(val[name], dtype=float)).reshape(-1)
        if entry.size == 1 and size > 1:
            entry = np.full(size, entry[0])
        if entry.size != size:
            raise ValueError(
                f"{label}[{name!r}] has {entry.size} values but the struct "
                f"slot expects {size}"
            )
        flat.extend(entry.tolist())

    return np.array(flat, dtype=float).reshape(-1, 1)
