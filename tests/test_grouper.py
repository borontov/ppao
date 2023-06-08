import numpy as np
import pytest
from hypothesis import Verbosity, example, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

import tests.custom_strategies as custom_st
from ppao import Grouper, exceptions, settings


@given(
    grouper=custom_st.grouper(),
    pipelines_1=st.data(),
    pipelines_2=st.data(),
)
@example(
    grouper=Grouper(
        settings_=settings.DEFAULT_SETTINGS,
    ),
    pipelines_1=[
        [1, 2, 3, 4, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [2, 2, 2, 0, 0],
        [5, 6, 7, 0, 0],
        [1, 0, 0, 0, 0],
    ],
    pipelines_2=None,
).via("grouper example case")
@hypothesis_settings(
    verbosity=Verbosity.verbose,
    max_examples=500,
)
def test_grouper_success(
    grouper: Grouper,
    pipelines_1,
    pipelines_2,
):
    pipelines_length = 0
    for pipelines in pipelines_1, pipelines_2:
        if pipelines is None:
            continue
        if not isinstance(pipelines, (list, np.ndarray)):
            pipelines = pipelines.draw(
                st.one_of(
                    (
                        st.lists(
                            st.one_of(
                                st.lists(
                                    st.integers(min_value=0, max_value=255),
                                    min_size=grouper.settings.pipeline_size_limit,
                                    max_size=grouper.settings.pipeline_size_limit,
                                ),
                            ),
                            min_size=1,
                        ),
                        custom_st.correct_pipelines_numpy_array(
                            pipeline_size_limit=grouper.settings.pipeline_size_limit
                        ),
                    )
                )
            )
        fail = False
        if isinstance(pipelines, (list, tuple)):
            if not pipelines or not all(
                len(x) == grouper.settings.pipeline_size_limit
                for x in pipelines
            ):
                fail = True

        elif isinstance(pipelines, np.ndarray) and (
            len(pipelines.shape) < 2
            or (
                pipelines.shape[1] != grouper.settings.pipeline_size_limit
                or pipelines.size == 0
            )
        ):
            fail = True
        if fail:
            with pytest.raises(
                (exceptions.PipelinesShapeError, exceptions.CreateArrayError)
            ):
                grouper.add(pipelines=pipelines)
            continue
        grouper.add(pipelines=pipelines)
        assert grouper.pipelines.size
        iterations = 0
        group = True
        while group is not None:
            group = grouper.pop()
            if group is not None:
                assert group is None or isinstance(group, np.ndarray)
                assert 1 <= group.shape[0] <= grouper.settings.group_size_limit
                assert group.shape[1] == grouper.settings.pipeline_size_limit
                pipelines_length -= group.shape[0]
                iterations += 1
                assert not grouper._total_counter
        if iterations == 0:
            if isinstance(pipelines, np.ndarray):
                if (
                    len(pipelines.shape) == len(grouper.pipelines.shape)
                    and pipelines.shape[1] == grouper.pipelines.shape[1]
                ):
                    pipelines_length += pipelines.shape[0]
            else:
                if all(
                    len(x) == grouper.settings.pipeline_size_limit
                    for x in pipelines
                ):
                    pipelines_length += len(pipelines)
            assert grouper.pipelines.shape[0] >= pipelines_length
            assert grouper._counters
            assert (
                not grouper._get_most_common_operations()
                or grouper.pipelines.shape[0] < 2
            )


@given(
    grouper=custom_st.grouper(),
    pipelines_1=st.data(),
    pipelines_2=st.data(),
)
@hypothesis_settings(
    verbosity=Verbosity.verbose,
    max_examples=500,
)
def test_grouper_fail(
    grouper: Grouper,
    pipelines_1,
    pipelines_2,
):
    pipelines_length = 0
    for pipelines in pipelines_1, pipelines_2:
        if pipelines is None:
            continue
        if not isinstance(pipelines, (list, tuple, np.ndarray)):
            pipelines = pipelines.draw(
                st.one_of(
                    st.lists(
                        st.lists(
                            st.integers(),
                        ),
                    ),
                    custom_st.grouper_random_pipelines_numpy_array(),
                )
            )
        fail = False
        if isinstance(pipelines, (list, tuple)):
            if not pipelines or not all(
                len(x) == grouper.settings.pipeline_size_limit
                for x in pipelines
            ):
                fail = True

        elif isinstance(pipelines, np.ndarray) and (
            len(pipelines.shape) < 2
            or (
                pipelines.shape[1] != grouper.settings.pipeline_size_limit
                or pipelines.size == 0
            )
        ):
            fail = True
        if fail:
            with pytest.raises(
                (exceptions.PipelinesShapeError, exceptions.CreateArrayError)
            ):
                grouper.add(pipelines=pipelines)
        else:
            try:
                grouper.add(pipelines=pipelines)
            except (exceptions.ArraysConcatError, exceptions.CreateArrayError):
                continue
        iterations = 0
        group = True
        while group is not None:
            group = grouper.pop()
            if group is not None:
                assert group is None or isinstance(group, np.ndarray)
                assert 1 <= group.shape[0] <= grouper.settings.group_size_limit
                assert group.shape[1] == grouper.settings.pipeline_size_limit
                pipelines_length -= group.shape[0]
                iterations += 1
                assert not grouper._total_counter

        if iterations == 0:
            if isinstance(pipelines, np.ndarray):
                if (
                    len(pipelines.shape) == len(grouper.pipelines.shape)
                    and pipelines.shape[1] == grouper.pipelines.shape[1]
                ):
                    pipelines_length += pipelines.shape[0]
            else:
                if all(
                    len(x) == grouper.settings.pipeline_size_limit
                    for x in pipelines
                ):
                    pipelines_length += len(pipelines)
            assert grouper.pipelines.shape[0] >= pipelines_length
            if (
                len(grouper._total_counter) == 1
                and grouper._total_counter.most_common(1)[0] == 0
            ):
                assert grouper._counters
            assert (
                not grouper._get_most_common_operations()
                or grouper.pipelines.shape[0] < 2
            )


@given(
    grouper=custom_st.grouper(),
    pipelines_1=st.data(),
    pipelines_2=st.data(),
)
@hypothesis_settings(
    verbosity=Verbosity.verbose,
    max_examples=500,
)
def test_grouper_bad_pipelines_type_fail(
    grouper: Grouper,
    pipelines_1,
    pipelines_2,
):
    for pipelines in pipelines_1, pipelines_2:
        if pipelines is None:
            continue
        if not isinstance(pipelines, (list, tuple, np.ndarray)):
            pipelines = pipelines.draw(
                custom_st.everything_except(
                    excluded_types=(list, tuple, np.ndarray)
                )
            )
        with pytest.raises(exceptions.GrouperError):
            grouper.add(pipelines=pipelines)


def test_grouper_remaining():
    settings_ = settings.Settings(pipeline_size_limit=5)
    grouper = Grouper(settings_=settings_)
    grouper.add([[1, 1, 3, 4, 5], [6, 7, 8, 9, 0]])
    assert grouper.pop() is None
    assert (grouper.pipelines == [[1, 1, 3, 4, 5], [6, 7, 8, 9, 0]]).all()
    grouper.add([[1, 0, 0, 0, 0]])
    assert grouper.pop() is None
    grouper.add([[1, 1, 1, 1, 0]])
    expected = [
        [1, 1, 1, 1, 0],
        [1, 1, 3, 4, 5],
        [1, 0, 0, 0, 0],
        [6, 7, 8, 9, 0],
    ]
    assert (grouper.pop() == expected).all()


def test_grouper_example_case():
    settings_ = settings.Settings(pipeline_size_limit=5)
    grouper = Grouper(settings_=settings_)
    grouper.add(
        [
            [1, 2, 3, 4, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 6, 7, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    assert (
        grouper.pop()
        == np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [2, 2, 2, 0, 0],
                [1, 2, 3, 4, 0],
            ],
            dtype=np.uint16,
        )
    ).all()
    assert (
        grouper.pop()
        == np.array(
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [5, 6, 7, 0, 0]],
            dtype=np.uint16,
        )
    ).all()
    assert grouper.pop() is None
    assert not grouper.pipelines.size
