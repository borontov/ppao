"""Reusable constants."""
from typing import Sequence

# The only acceptable dtypes of ppao matrix.
ACCEPTABLE_DEFAULT_DTYPE: Sequence[str] = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)

# The only acceptable type codes of arrays.
ACCEPTABLE_ARRAY_TYPE_CODE: Sequence[str] = (
    "I",
    "H",
    "L",
    "Q",
)

# The only acceptable shift dtypes.
ACCEPTABLE_DEFAULT_SHIFT_ARRAY_DTYPE: Sequence[str] = (
    "int8",
    "int16",
)
