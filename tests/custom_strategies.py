from collections import Counter, defaultdict, namedtuple
from contextlib import suppress
from typing import Optional, Set, Type

import hypothesis.extra.numpy as np_st
import numpy as np
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy, composite, from_type

from ppao import Grouper, constants, settings
from ppao.custom_types import Frequency

horizontal_sequence_data = namedtuple(
    "horizontal_sequence_data",
    field_names=("settings", "sequence", "mapping"),
)


def everything_except(excluded_types: Type) -> SearchStrategy:
    """Hypothesis strategy to generate all possible data except excluded_types.
    :param excluded_types: types to exclude
    :return: hypothesis strategy
    """
    return (
        from_type(type)
        .flatmap(from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


@composite
def correct_settings(
    draw,
    default_dtype: Optional[str] = None,
    common_ops_percent_bound: Optional[float] = None,
    common_ops_bound: Optional[int] = None,
    group_size_limit: Optional[int] = None,
    pipeline_size_limit: Optional[int] = None,
    default_array_type_code: Optional[str] = None,
    default_shift_array_type_code: Optional[str] = None,
) -> settings.Settings:
    settings_ = settings.Settings(
        default_dtype=default_dtype
        or draw(st.sampled_from(constants.ACCEPTABLE_DEFAULT_DTYPE)),
        common_ops_percent_bound=common_ops_percent_bound
        or draw(st.floats(min_value=0.01, max_value=0.99)),
        common_ops_bound=common_ops_bound
        or draw(st.integers(min_value=1, max_value=3)),
        group_size_limit=group_size_limit
        or draw(st.integers(min_value=2, max_value=7)),
        pipeline_size_limit=pipeline_size_limit
        or draw(st.integers(min_value=2, max_value=4)),
        default_array_type_code=default_array_type_code
        or draw(st.sampled_from(constants.ACCEPTABLE_ARRAY_TYPE_CODE)),
        default_shift_array_dtype=default_shift_array_type_code
        or draw(
            st.sampled_from(constants.ACCEPTABLE_DEFAULT_SHIFT_ARRAY_DTYPE)
        ),
    )
    return settings_


@composite
def grouper(draw, **kwargs) -> Grouper:
    grouper_settings = draw(correct_settings(**kwargs))
    return Grouper(settings_=grouper_settings)


@composite
def correct_pipelines_numpy_array(
    draw,
    pipeline_size_limit,
    max_rows: Optional[int] = 7,
    unique: bool = False,
) -> np.ndarray:
    correct_array = draw(
        np_st.arrays(
            dtype=np_st.unsigned_integer_dtypes(),
            shape=st.tuples(
                st.integers(min_value=1, max_value=max_rows),
                st.just(pipeline_size_limit),
            ),
            elements=st.integers(min_value=1, max_value=255),
            unique=unique,
        )
    )
    return correct_array


@composite
def grouper_random_pipelines_numpy_array(
    draw,
) -> np.ndarray:
    random_array = draw(
        np_st.arrays(
            dtype=np_st.array_dtypes(),
            shape=np_st.array_shapes(),
        )
    )
    return random_array


@composite
def correct_horizontal_sequence(
    draw,
    **kwargs,
) -> horizontal_sequence_data:
    horizontal_optimizer_settings: settings.Settings = draw(
        correct_settings(**kwargs)
    )
    arrays_range = range(
        draw(
            st.integers(
                min_value=1,
                max_value=horizontal_optimizer_settings.group_size_limit,
            )
        )
    )
    horizontal_sequence = tuple(
        list(
            draw(
                st.sets(
                    elements=st.integers(min_value=0, max_value=100),
                    min_size=1,
                    max_size=horizontal_optimizer_settings.pipeline_size_limit,
                )
            )
        )
        for _ in arrays_range
    )
    mapping = defaultdict(lambda: defaultdict(list))
    for array_key, array in enumerate(horizontal_sequence):
        for item in array:
            unique_pipeline_ids = draw(
                st.sets(
                    elements=st.integers(
                        min_value=0,
                        max_value=horizontal_optimizer_settings.group_size_limit,
                    ),
                    min_size=1,
                    max_size=horizontal_optimizer_settings.group_size_limit,
                )
            )
            mapping[array_key][item].extend(list(unique_pipeline_ids))

    return horizontal_sequence_data(
        settings=horizontal_optimizer_settings,
        sequence=horizontal_sequence,
        mapping=mapping,
    )


@composite
def from_array_incorrect_dtype_random_shape(draw):
    shape = draw(np_st.array_shapes())
    dtype = draw(
        np_st.array_dtypes().filter(
            lambda dtype: str(dtype) not in constants.ACCEPTABLE_DEFAULT_DTYPE
        ),
    )
    from_array = draw(
        np_st.arrays(
            shape=shape,
            dtype=dtype,
        )
    )
    return from_array


@composite
def frequency_random_correct(draw):
    total = draw(st.integers(min_value=5, max_value=20))
    most_common = draw(
        st.sets(
            elements=st.integers(min_value=1),
            min_size=1,
            max_size=3,
        )
    )
    return Frequency(total=total, most_common=most_common)


def frequency(pipelines, settings_) -> Optional[Frequency]:
    total_counter = Counter()
    if not pipelines.size:
        return
    for pipeline in pipelines:
        operation_frequency_counter = Counter()
        for operation_index, row_operation_frequency in zip(
            *np.unique(ar=pipeline, return_counts=True),
            strict=True,
        ):
            if operation_index != 0:
                operation_frequency_counter[
                    operation_index
                ] += row_operation_frequency
        total_counter += operation_frequency_counter
    with suppress(KeyError):
        # ignore operation indexes equal to zero
        del total_counter[0]
    most_common_operations: Set[int] = set()
    operation_frequency_sum = 0
    for (
        common_operation,
        operation_frequency,
    ) in total_counter.most_common(settings_.group_size_limit):
        operation_frequency_sum += operation_frequency
        most_common_operations.add(common_operation)
        if (
            operation_frequency_sum / total_counter.total()
            >= settings_.common_ops_percent_bound
        ):
            break
        if len(most_common_operations) >= settings_.common_ops_bound:
            break

    if (
        not total_counter
        or operation_frequency_sum / total_counter.total()
        < settings_.common_ops_percent_bound
    ):
        return
    return Frequency(
        most_common=most_common_operations, total=total_counter.total()
    )
