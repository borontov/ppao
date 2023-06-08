from array import array
from collections import defaultdict
from functools import partial
from typing import Dict, Generator, List, Set, Tuple, Union

import numpy as np

from ppao import exceptions, settings
from ppao.custom_types import Frequency


class SourceMatrix(np.ndarray):
    """Matrix of operation pipelines.

    Attributes:
        total_operations: total number of operations (not unique).
        most_common: most common operations represented in the matrix.
        settings: ppao settings.
    """

    __slots__ = (
        "total_operations",
        "most_common",
        "settings",
    )

    def __new__(
        cls,
        from_array: np.ndarray,
        frequency: Frequency,
        settings_: settings.Settings = settings.DEFAULT_SETTINGS,
    ) -> np.ndarray:
        """Allows to inherit from np.ndarray class.

        :param from_array: pipeline array.
        """
        cls._validate_input(from_array, frequency, settings_)
        obj = np.asarray(from_array).view(cls)
        return obj

    @classmethod
    def _validate_input(
        cls,
        from_array: np.ndarray,
        frequency: Frequency,
        settings_: settings.Settings,
    ) -> None:
        if (
            (not isinstance(from_array, np.ndarray))
            or (not isinstance(settings_, settings.Settings))
            or (not isinstance(frequency, Frequency))
        ):
            raise exceptions.MatrixAttributeTypeValidationError()
        if (
            len(from_array.shape) == 1
            or from_array.shape[1] != settings_.pipeline_size_limit
        ):
            raise exceptions.ArrayShapeError(
                shape=(
                    f"* <= {settings_.group_size_limit}",
                    settings_.pipeline_size_limit,
                ),
            )
        if not np.issubdtype(from_array.dtype, np.unsignedinteger):
            raise exceptions.ArrayDtypeError(dtype=from_array.dtype)

    def __init__(
        self,
        from_array: np.ndarray,
        frequency: Frequency,
        settings_: settings.Settings = settings.DEFAULT_SETTINGS,
    ) -> None:
        self.settings = settings_
        self.most_common = np.fromiter(
            (x for x in frequency.most_common), dtype=settings_.default_dtype
        )
        self.total_operations = frequency.total

    def get_windows(self, operation: int) -> np.ndarray:
        return np.fromiter(
            self.window_generator(operation),
            dtype=[
                ("start", self.settings.default_shift_array_dtype),
                ("end", self.settings.default_shift_array_dtype),
                ("size", self.settings.default_shift_array_dtype),
                ("key", self.settings.default_shift_array_dtype),
            ],
        )

    def window_generator(self, operation: int) -> Generator:
        """Get all windows from matrix rows by the operation id.

        :param operation: operation id.
        :return: generator of the window sequence.
        """
        for key, row in enumerate(self):
            indexes = np.where(row == operation)[0]
            if indexes.size >= 1:
                start = indexes[0]
                end = indexes[-1]
                size = end - start + 1
                yield (
                    start,
                    end,
                    size,
                    key,
                )
            else:
                yield 0, 0, 0, key

    def get_window_sizes_delta_sequence(
        self,
        windows: np.ndarray,
    ) -> np.ndarray:
        """
        :return: the sequence of differences between window sizes.
        """
        longest_window_index = np.unravel_index(
            np.argmax(windows["size"]), windows.shape
        )[0]
        longest_window = windows[[longest_window_index]]
        func = partial(self.handle_window, longest_window=longest_window)
        result = np.array(
            tuple(
                map(
                    func,
                    windows,
                )
            ),
        )
        return result

    def handle_window(
        self, window: np.ndarray, longest_window: np.ndarray
    ) -> np.ndarray:
        if not window["size"]:
            output = (0, 0)
        else:
            output = (
                int(longest_window["size"]) - int(window["size"]),
                int(longest_window["start"]) - int(window["start"]),
            )
        return np.array(
            output,
            dtype=[
                ("delta", self.settings.default_shift_array_dtype),
                ("offset", self.settings.default_shift_array_dtype),
            ],
        )

    def get_possible_shifts(self, delta_offset: np.ndarray) -> Generator:
        """
        :param window_sizes_delta_sequence: a sequence of differences between
        window sizes.
        :return: possible shifts bounded by operation windows.
        """
        for row in delta_offset:
            offset = row["offset"]
            delta = row["delta"]
            yield tuple(range(offset, offset + delta + 1))

    def get_all_combinations(
        self, possible_shifts: Generator[Tuple[int, ...], None, None]
    ) -> np.ndarray:
        return np.array(
            np.meshgrid(*possible_shifts),
            dtype=self.settings.default_shift_array_dtype,
        ).T.reshape(-1, self.shape[0])

    def count_result(self, shifts, best_result):
        horizontal_sequence = self.make_horizontal_sequence(
            shifts=shifts, for_counter=True
        )
        result = sum((len(column) for column in horizontal_sequence))
        if result < best_result["result"]:
            best_result["result"] = result
            best_result["shifts"] = shifts

    def make_mapping(self):
        column_key = 0
        mapping: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for column in self.T:
            if any(column):
                for pipeline_id, operation in enumerate(column):
                    if operation != 0:
                        mapping[column_key][operation].append(pipeline_id)
                column_key += 1
        return mapping

    def make_horizontal_sequence(
        self,
        shifts: np.ndarray,
        for_counter: bool = False,
    ) -> Union[
        List[Set[int]],
        Tuple[Tuple[array, ...], Dict[int, Dict[int, List[int]]]],
    ]:
        if not for_counter:
            mapping: Dict[int, Dict[int, List[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
        results = []
        last_column = None
        min_shift = min(shifts)
        for (
            column_index,
            row_index,
            item_index,
        ) in self._iterate_shifted_matrix(shifts=shifts):
            operation = self[row_index][item_index]
            if not for_counter:
                mapping[column_index - min_shift][operation].append(row_index)
            if last_column != column_index:
                last_column = column_index
                results.append(set())
            results[-1].add(operation)
        if not for_counter:
            results = tuple(
                array(self.settings.default_array_type_code, unit)
                for unit in results
            )
            return results, mapping
        return results

    def _iterate_shifted_matrix(
        self,
        shifts: np.ndarray,
    ) -> Generator[Tuple[int, int, int], None, None]:
        operations_in_pipeline = self.shape[1]
        min_shift = min(shifts)
        column_rows: Dict[int, Set[int]] = defaultdict(set)
        for row_index in range(len(shifts)):
            shift = shifts[row_index]
            for column_index in range(shift, operations_in_pipeline + shift):
                column_rows[column_index].add(row_index)
        for column_index in range(
            min_shift, operations_in_pipeline + max(shifts)
        ):
            for row_index in column_rows[column_index]:
                shift = shifts[row_index]
                item_index = column_index - shift
                yield column_index, row_index, item_index
