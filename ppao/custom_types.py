"""Classes for data validation."""
import collections
import dataclasses
from typing import Any, Sequence, Set

import numpy as np

from ppao import exceptions


@dataclasses.dataclass(slots=True)
class ExecutionUnit:
    """Pipeline ids connected with operation id.

    Attributes:
        operation: operation id.
        pipelines: ordered sequence of pipeline ids of that operation.
    """

    operation: int
    pipelines: np.ndarray

    def __post_init__(self) -> None:
        """Attribute validation."""
        if (not isinstance(self.operation, int)) or (
            not isinstance(self.pipelines, np.ndarray)
        ):
            raise exceptions.CustomTypeAttributeTypeValidationError()

        if self.pipelines.size == 0:
            raise exceptions.CustomTypeEmptyArrayError()


@dataclasses.dataclass(slots=True, frozen=True)
class Frequency:
    """Operation frequency. Zero is not counted.

    Attributes:
        total: total number of operations.
        most_common: the most frequent operations bounded by settings.
    """

    total: int
    most_common: Set[int]

    def __post_init__(self) -> None:
        """Attribute validation."""
        if (not isinstance(self.most_common, set)) or (
            not np.issubdtype(type(self.total), np.integer)
        ):
            raise exceptions.CustomTypeAttributeTypeValidationError()
        if self.total <= 0 or any(
            i <= 0 or not np.issubdtype(type(i), np.integer)
            for i in self.most_common
        ):
            raise exceptions.CustomTypeAttributeTypeValidationError()
        if not self.most_common:
            raise exceptions.MostCommonIsEmptyError()


class Solution(collections.UserList):
    """List of ExecutionUnit elements.

    This list has been composed in the true order of operations execution.

    Attributes:
        shifts: an array of shifts of the solution matrix.
        result: number of ExecutionUnit elements.
    """

    __slots__ = (
        "shifts",
        "result",
    )

    def __init__(
        self,
        execution_units: Sequence[ExecutionUnit],
        shifts: np.ndarray,
        result: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shifts = shifts
        self.result = result
        self._validation()
        self.extend(execution_units)

    def _validation(self) -> None:
        """Attribute validation."""
        if (not isinstance(self.shifts, np.ndarray)) or (
            not isinstance(self.result, int)
        ):
            raise exceptions.CustomTypeAttributeTypeValidationError()
        if self.shifts.size == 0:
            raise exceptions.CustomTypeEmptyArrayError()

    def check(self, value: ExecutionUnit) -> None:
        """Type checking."""
        if not isinstance(value, ExecutionUnit):
            raise exceptions.CustomTypeAttributeTypeValidationError()

    def __setitem__(self, index_: int, value: ExecutionUnit) -> None:
        self.check(value)
        self.data[index_] = value

    def insert(self, index_: int, value: ExecutionUnit) -> None:
        self.check(value)
        self.data.insert(index_, value)

    def append(self, value: ExecutionUnit) -> None:
        self.check(value)
        self.data.append(value)

    def extend(self, values: Sequence[ExecutionUnit]) -> None:
        for value in values:
            self.check(value)
            self.data.append(value)
