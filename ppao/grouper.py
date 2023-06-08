from collections import Counter, defaultdict
from contextlib import suppress
from typing import DefaultDict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ppao import exceptions, settings
from ppao.custom_types import Frequency
from ppao.matrix import SourceMatrix


class Grouper:
    """Preprocessing input data to create correct groups for the solver.

    Attributes:
        settings: ppao settings.
        pipelines: remaining pipelines after add() and pop() calls.
    """

    def __init__(
        self,
        settings_: settings.Settings = settings.DEFAULT_SETTINGS,
    ) -> None:
        self.settings = settings_
        self.pipelines = np.empty(
            dtype=self.settings.default_dtype,
            shape=(0, self.settings.pipeline_size_limit),
        )
        self._counters: DefaultDict[int, Counter] = defaultdict(Counter)
        self._total_counter = Counter()

    def _clear(self, most_common_scores: List[Tuple[int, int]]) -> None:
        self._total_counter.clear()
        keys_to_delete = tuple(
            row_index for row_index, _ in most_common_scores
        )
        for key in keys_to_delete:
            del self._counters[key]
        self.pipelines = np.delete(
            arr=self.pipelines, obj=keys_to_delete, axis=0
        )

    def _count_frequency(self) -> bool:
        if not self.pipelines.size:
            return False
        for row_index, pipeline in enumerate(self.pipelines):
            operation_frequency_counter: Counter = Counter()
            for operation_index, row_operation_frequency in zip(
                *np.unique(ar=pipeline, return_counts=True), strict=True
            ):
                if operation_index != 0:
                    operation_frequency_counter[
                        operation_index
                    ] += row_operation_frequency
            self._counters[row_index] = operation_frequency_counter
            self._total_counter += operation_frequency_counter
        with suppress(KeyError):
            # ignore operation indexes equal to zero
            del self._total_counter[0]
        if not self._total_counter:
            return False
        return True

    def pop(self) -> Optional[SourceMatrix]:
        """Get a group and remove it from grouper."""
        if not self._count_frequency():
            return
        if self.pipelines.shape[0] < 2:
            return
        if len(self._total_counter) == 1:
            most_common_operations = set(x for x in self._total_counter.keys())
        else:
            most_common_operations = self._get_most_common_operations()
        biggest_scores = self._get_biggest_acceptance_scores(
            most_common_operations
        )
        if not biggest_scores:
            return
        group = [self.pipelines[row_key] for row_key, score in biggest_scores]
        if group:
            pipelines = np.array(group, dtype=self.settings.default_dtype)
            frequency = Frequency(
                total=int(self._total_counter.total()),
                most_common=most_common_operations,
            )
            self._clear(biggest_scores)
            return SourceMatrix(  # pytype: disable=bad-return-type
                from_array=pipelines,
                frequency=frequency,
                settings_=self.settings,
            )

    def _get_biggest_acceptance_scores(
        self, most_common_operations: Set[int]
    ) -> Optional[List[Tuple[int, int]]]:
        scores: Counter = Counter()
        for row_index in range(len(self.pipelines)):
            for common_operation in most_common_operations:
                scores[row_index] += (
                    self._counters[row_index].get(common_operation, 0)
                    * self._total_counter[common_operation]
                )
        biggest_scores = scores.most_common(self.settings.group_size_limit)
        if not biggest_scores:
            return None
        return biggest_scores

    def _get_most_common_operations(self) -> Set[int]:
        most_common_operations: Set[int] = set()
        operation_frequency_sum = 0
        for (
            common_operation,
            operation_frequency,
        ) in self._total_counter.most_common(self.settings.group_size_limit):
            operation_frequency_sum += operation_frequency
            most_common_operations.add(common_operation)
            if self._stop_condition(
                operation_frequency_sum, most_common_operations
            ):
                break
        if (
            not self._total_counter
            or operation_frequency_sum / self._total_counter.total()
            < self.settings.common_ops_percent_bound
        ):
            return set()
        return most_common_operations

    def _stop_condition(
        self,
        operation_frequency_sum: int,
        most_common_operations: Set[int],
    ) -> bool:
        if (
            operation_frequency_sum / self._total_counter.total()
            >= self.settings.common_ops_percent_bound
        ):
            return True
        if len(most_common_operations) >= self.settings.common_ops_bound:
            return True
        return False

    def add(self, pipelines: Union[Sequence, np.ndarray]) -> None:
        """Add pipelines to the grouper."""
        pipelines_array = self._validate_pipelines_and_create_array(pipelines)
        self._concatenate_arrays(pipelines_array)

    def _concatenate_arrays(self, pipelines: np.ndarray) -> None:
        try:
            self.pipelines = np.concatenate((self.pipelines, pipelines))
        except (ValueError, TypeError):
            raise exceptions.ArraysConcatError(  # noqa: B904
                array_1=self.pipelines, array_2=pipelines
            )

    def _validate_pipelines_and_create_array(
        self, pipelines: Union[tuple, list, np.ndarray]
    ) -> np.ndarray:
        if not isinstance(pipelines, (tuple, list, np.ndarray)):
            raise exceptions.CreateArrayError()
        if (
            isinstance(pipelines, np.ndarray)
            and (
                len(pipelines.shape) < 2
                or (
                    pipelines.shape[1] != self.settings.pipeline_size_limit
                    or pipelines.size == 0
                )
            )
        ) or (
            isinstance(pipelines, (list, tuple))
            and (
                not pipelines
                or not all(
                    len(x) == self.settings.pipeline_size_limit
                    for x in pipelines
                )
            )
        ):
            raise exceptions.PipelinesShapeError(
                length=self.pipelines.shape[1]
            )
        try:
            pipelines = np.array(pipelines, dtype=self.pipelines.dtype)
            return pipelines
        except (OverflowError, TypeError):
            raise exceptions.CreateArrayError()  # noqa: B904
