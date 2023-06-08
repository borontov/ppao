import copy
from array import array
from collections import defaultdict
from contextlib import suppress
from typing import Dict, Generator, List, Optional, Set, Tuple

import numpy as np

from ppao import ExecutionUnit, Solution, exceptions, settings
from ppao.matrix import SourceMatrix


class PipelineMatrixSolver:
    """Algorithmic solver of the pipeline optimization problem.

    Attributes:
        source_matrix: pipelines matrix array.
    """

    __slots__ = (
        "source_matrix",
        "settings",
    )

    def __init__(
        self,
        source_matrix: SourceMatrix,
        settings_: settings.Settings = settings.DEFAULT_SETTINGS,
    ) -> None:
        self.source_matrix = source_matrix
        self.settings = settings_

    def solve(self) -> Solution:
        """
        :return: the problem solution.
        """
        if self.source_matrix.most_common.size == 0:
            raise exceptions.MostCommonIsEmptyError()
        most_common_operations = self.source_matrix.most_common
        possible_shifts: Dict[int, Set[int]] = defaultdict(set)
        for common_operation in most_common_operations:
            windows = self.source_matrix.get_windows(common_operation)
            delta_offset = self.source_matrix.get_window_sizes_delta_sequence(
                windows
            )
            for i, shifts in enumerate(
                self.source_matrix.get_possible_shifts(delta_offset)
            ):
                possible_shifts[i].update(shifts)
        all_combinations = self.source_matrix.get_all_combinations(
            tuple(shifts) for shifts in possible_shifts.values()
        )
        best_result = {"result": np.inf}
        tuple(
            self.source_matrix.count_result(combination, best_result)
            for combination in all_combinations
        )
        sequence, mapping = self.source_matrix.make_horizontal_sequence(
            shifts=best_result["shifts"],
        )
        horizontal_optimizer = HorizontalOptimizer(
            source_sequence=sequence,
        )
        execution_units = horizontal_optimizer.optimize(mapping=mapping)
        solution = Solution(
            execution_units=execution_units,
            shifts=best_result["shifts"],
            result=best_result["result"],
        )
        return solution


class HorizontalOptimizer:
    """Final optimization of the solver solution."""

    def __init__(
        self,
        source_sequence: Tuple[array, ...],
        settings_: settings.Settings = settings.DEFAULT_SETTINGS,
    ) -> None:
        self.source_sequence = source_sequence
        self.sequence_length = len(self.source_sequence)
        self.settings = settings_
        self.sorted_keys: Set[int] = set()
        self.sorted_parts: Dict[int, Dict[int, int]] = defaultdict(dict)

    def _register(self, key: int, index_before: int, index_after: int) -> None:
        if index_after in self.sorted_parts[key].values():
            return
        for index in (key, index_before, index_after):
            if not isinstance(index, int) or index < 0:
                raise exceptions.IndexValidationError()
        self.sorted_parts[key][index_before] = index_after
        last_index = self._get_last_index(key)
        one_or_less_unsorted = len(self.sorted_parts[key]) >= last_index
        sides_sorted = self.sorted_parts[key].get(0) and self.sorted_parts[
            key
        ].get(last_index)
        if one_or_less_unsorted or sides_sorted:
            self.sorted_keys.add(key)

    def _get_sort_order(self) -> Generator[List[int], None, None]:
        reversed_sorted_parts = self._get_reversed_sorted_parts()
        for array_key in range(self.sequence_length):
            sort_order = dict()
            unsorted_items = self._get_unsorted(array_key)
            for index_after_sort in range(
                len(self.source_sequence[array_key])
            ):
                if index_after_sort in self.sorted_parts[array_key].values():
                    index_before_sort = reversed_sorted_parts[array_key][
                        index_after_sort
                    ]
                    operation = self.source_sequence[array_key][
                        index_before_sort
                    ]
                    sort_order[index_after_sort] = operation
                else:
                    if unsorted_items:
                        sort_order[index_after_sort] = unsorted_items.pop()
            sorted_tuple = sorted(sort_order.items(), key=lambda x: x[0])
            yield list(x[1] for x in sorted_tuple)

    def _get_reversed_sorted_parts(self) -> Dict[int, Dict[int, int]]:
        return {
            key: {
                index_after_sort: index_before_sort
                for index_before_sort, index_after_sort in dict_.items()
            }
            for key, dict_ in self.sorted_parts.items()
        }

    def optimize(
        self,
        mapping: Dict[int, Dict[int, List[int]]],
    ) -> Tuple[ExecutionUnit]:
        self._sort_singles()
        self._sort_both_sides()
        self._sort_one_side()
        sort_order = self._get_sort_order()
        execution_units = tuple(
            self._get_execution_units(sort_order=sort_order, mapping=mapping)
        )
        return execution_units

    def _get_execution_units(
        self,
        sort_order: Generator[List[int], None, None],
        mapping: Dict[int, Dict[int, List[int]]],
    ) -> Generator[ExecutionUnit, None, None]:
        last_operation = None
        pipelines = list()
        for sort_mapping, pipelines_mapping in zip(
            sort_order,
            tuple(mapping.values()),
            strict=True,
        ):
            for operation in sort_mapping:
                if last_operation != operation:
                    if last_operation is not None:
                        yield self._make_execution_unit(
                            last_operation,
                            pipelines,
                        )
                        pipelines = list()
                    last_operation = operation
                pipelines.extend(pipelines_mapping[operation])
        if pipelines and last_operation is not None:
            yield self._make_execution_unit(last_operation, pipelines)

    def _make_execution_unit(
        self, operation: int, pipelines: List[int]
    ) -> ExecutionUnit:
        return ExecutionUnit(
            operation=operation,
            pipelines=np.array(
                copy.deepcopy(pipelines),
                dtype=self.settings.default_dtype,
            ),
        )

    def _sort_both_sides(self) -> None:
        for key in range(self.sequence_length):
            left_key, right_key = self._get_left_and_right_key(key)
            if None in (left_key, right_key) or self.sorted_keys.intersection(
                {key, left_key, right_key}
            ):
                continue
            left_unsorted, current_unsorted, right_unsorted = (
                self._get_unsorted(left_key),
                self._get_unsorted(key),
                self._get_unsorted(right_key),
            )
            left_intersection = array(
                self.settings.default_array_type_code,
                current_unsorted.intersection(left_unsorted),
            )
            right_intersection = array(
                self.settings.default_array_type_code,
                current_unsorted.intersection(right_unsorted),
            )
            if not left_intersection or not right_intersection:
                continue
            left_intersection_length, right_intersection_length = len(
                left_intersection
            ), len(right_intersection)
            if left_intersection_length == right_intersection_length == 1:
                if left_intersection == right_intersection:
                    continue
                else:
                    left_chosen_item = left_intersection[0]
                    right_chosen_item = right_intersection[0]
            elif right_intersection_length == 1 < left_intersection_length:
                right_chosen_item = right_intersection[0]
                with suppress(ValueError):
                    left_intersection.remove(right_chosen_item)
                left_chosen_item = left_intersection[0]
            else:
                left_chosen_item = left_intersection[0]
                with suppress(ValueError):
                    right_intersection.remove(left_chosen_item)
                right_chosen_item = right_intersection[0]
            self._move_right(left_key, left_chosen_item)
            self._move_right(key, right_chosen_item)
            self._move_left(key, left_chosen_item)
            self._move_left(right_key, right_chosen_item)

    def _move_left(self, key: int, item: int) -> None:
        self._register(
            key,
            index_before=self._get_index(key, item),
            index_after=0,
        )

    def _move_right(self, key: int, item: int) -> None:
        self._register(
            key,
            index_before=self._get_index(key, item),
            index_after=self._get_last_index(key),
        )

    def _sort_one_side(self) -> None:
        for current_key in range(self.sequence_length):
            left_key, right_key = self._get_left_and_right_key(current_key)
            current_unsorted = self._get_unsorted(current_key)
            for side_key in left_key, right_key:
                if (
                    not self.sorted_keys.intersection({current_key, side_key})
                    or not side_key
                ):
                    continue
                side_unsorted = self._get_unsorted(side_key)
                side_intersection = array(
                    self.settings.default_array_type_code,
                    current_unsorted.intersection(side_unsorted),
                )
                if not side_intersection:
                    continue

                side_chosen_item = side_intersection[0]
                if side_key == right_key:
                    move_left_key = side_key
                    move_right_key = current_key
                else:
                    move_left_key = current_key
                    move_right_key = side_key
                self._move_left(move_left_key, side_chosen_item)
                self._move_right(move_right_key, side_chosen_item)

    def _sort_singles(self) -> None:
        for key in range(self.sequence_length):
            if len(self.source_sequence[key]) == 1:
                self._sort_single(key)

    def _sort_single(self, key: int) -> None:
        self._register(key=key, index_before=0, index_after=0)
        left_key, right_key = self._get_left_and_right_key(key)
        single_operation_id = self.source_sequence[key][0]
        if (
            left_key is not None
            and single_operation_id in self.source_sequence[left_key]
        ):
            self._move_right(left_key, single_operation_id)
        if (
            right_key is not None
            and single_operation_id in self.source_sequence[right_key]
        ):
            self._move_left(right_key, single_operation_id)

    def _get_index(self, key: int, value: int) -> Optional[int]:
        try:
            return self.source_sequence[key].index(value)
        except ValueError:
            return None

    def _get_last_index(self, key: int) -> int:
        return len(self.source_sequence[key]) - 1

    def _get_left_and_right_key(
        self,
        key: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        if key == 0:
            left = None
        else:
            left = key - 1
        if self.sequence_length == key + 1:
            right = None
        else:
            right = key + 1
        return left, right

    def _get_unsorted(self, key: int) -> Set:
        unsorted = set()
        for i in range(len(self.source_sequence[key])):
            if i not in self.sorted_parts[key]:
                unsorted.add(self.source_sequence[key][i])
        return unsorted
