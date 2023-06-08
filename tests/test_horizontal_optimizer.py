import random
from datetime import timedelta

from hypothesis import Verbosity, given
from hypothesis import settings as hypothesis_settings

import ppao
import tests.custom_strategies as custom_st


@given(
    data=custom_st.correct_horizontal_sequence(),
)
@hypothesis_settings(
    verbosity=Verbosity.verbose,
    max_examples=100,
    deadline=timedelta(seconds=2),
)
def test_horizontal_optimizer(data):
    optimizer = ppao.solver.HorizontalOptimizer(
        source_sequence=data.sequence, settings_=data.settings
    )
    sequence_size = sum(len(x) for x in data.sequence)
    result = optimizer.optimize(mapping=data.mapping)
    assert len(result) <= sequence_size


@given(
    data=custom_st.correct_horizontal_sequence(),
)
def test_get_last_index(data):
    optimizer = ppao.solver.HorizontalOptimizer(
        source_sequence=data.sequence, settings_=data.settings
    )
    key = random.randint(0, len(data.sequence) - 1)
    assert optimizer._get_last_index(key) == (len(data.sequence[key]) - 1)


@given(
    data=custom_st.correct_horizontal_sequence(),
)
def test_register(data):
    optimizer = ppao.solver.HorizontalOptimizer(
        source_sequence=data.sequence, settings_=data.settings
    )
    key = random.randint(0, len(data.sequence) - 1)
    index_before = random.randint(0, len(data.sequence[key]) - 1)
    index_after = random.randint(0, len(data.sequence[key]) - 1)
    optimizer._register(
        key=key, index_before=index_before, index_after=index_after
    )
    assert optimizer.sorted_parts[key][index_before] == index_after
    last_index = optimizer._get_last_index(key)
    one_or_less_unsorted = len(optimizer.sorted_parts[key]) >= last_index
    start_and_end_sorted = optimizer.sorted_parts[key].get(
        0
    ) and optimizer.sorted_parts[key].get(last_index)
    if one_or_less_unsorted or start_and_end_sorted:
        assert key in optimizer.sorted_keys


@given(
    data=custom_st.correct_horizontal_sequence(),
)
def test_get_unsorted(data):
    optimizer = ppao.solver.HorizontalOptimizer(
        source_sequence=data.sequence,
        settings_=data.settings,
    )
    key = random.randint(0, len(data.sequence) - 1)
    index_before = random.randint(0, len(data.sequence[key]) - 1)
    index_after = random.randint(0, len(data.sequence[key]) - 1)
    optimizer._register(
        key=key,
        index_before=index_before,
        index_after=index_after,
    )
    assert data.sequence[key][index_before] not in optimizer._get_unsorted(
        key=key,
    )


@given(
    data=custom_st.correct_horizontal_sequence(),
)
def test_get_left_and_right_key(data):
    optimizer = ppao.solver.HorizontalOptimizer(
        source_sequence=data.sequence, settings_=data.settings
    )
    key = random.randint(0, len(data.sequence) - 1)
    if key == 0:
        left = None
    else:
        left = key - 1
    if len(data.sequence) == key + 1:
        right = None
    else:
        right = key + 1
    assert (left, right) == optimizer._get_left_and_right_key(key)
