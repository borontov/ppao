from datetime import timedelta

import numpy as np
import pytest
from hypothesis import Verbosity, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

import ppao
import tests.custom_strategies as custom_st
from ppao import ExecutionUnit, exceptions, settings


@given(
    settings_=custom_st.correct_settings(),
    pipelines_1=st.data(),
    pipelines_2=st.data(),
)
@hypothesis_settings(
    verbosity=Verbosity.verbose,
    max_examples=1000,
    deadline=timedelta(seconds=1),
)
def test_solver(
    settings_: settings.Settings,
    pipelines_1,
    pipelines_2,
):
    if not isinstance(pipelines_1, np.ndarray):
        pipelines_1 = pipelines_1.draw(
            custom_st.correct_pipelines_numpy_array(
                pipeline_size_limit=settings_.pipeline_size_limit,
                max_rows=settings_.group_size_limit,
            )
        )
    if not isinstance(pipelines_2, np.ndarray):
        pipelines_2 = pipelines_2.draw(
            custom_st.correct_pipelines_numpy_array(
                pipeline_size_limit=settings_.pipeline_size_limit,
                max_rows=settings_.group_size_limit,
            )
        )
    solver = ppao.PipelineMatrixSolver(
        source_matrix=pipelines_1,
        settings_=settings_,
    )
    for pipelines in (pipelines_1, pipelines_2):
        frequency = custom_st.frequency(
            pipelines=pipelines,
            settings_=settings_,
        )
        if frequency is None:
            with pytest.raises(exceptions.MatrixAttributeTypeValidationError):
                ppao.SourceMatrix(
                    from_array=pipelines,
                    settings_=settings_,
                    frequency=frequency,
                )
            return
        operations_pipeline_matrix = ppao.SourceMatrix(
            from_array=pipelines,
            settings_=settings_,
            frequency=frequency,
        )

        solver.source_matrix = operations_pipeline_matrix
        if not operations_pipeline_matrix.most_common.size:
            with pytest.raises(exceptions.MostCommonIsEmptyError):
                solver.solve()
            continue
        else:
            solution = solver.solve()
        assert len(solution) <= pipelines.size
        assert len(solution.shifts) == pipelines.shape[0]
        assert solution.result
        assert len(solution) <= solution.result
        assert (
            sum(array_.pipelines.size for array_ in solution) == pipelines.size
        )


@given(
    grouper=custom_st.grouper(),
    pipelines_1=st.data(),
    pipelines_2=st.data(),
)
@hypothesis_settings(
    verbosity=Verbosity.verbose,
    max_examples=1000,
    deadline=timedelta(seconds=1),
)
def test_solver_and_grouper(
    grouper: ppao.grouper.Grouper,
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
                            pipeline_size_limit=grouper.settings.pipeline_size_limit,
                        ),
                    )
                )
            )
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
                solver = ppao.PipelineMatrixSolver(
                    source_matrix=group,
                    settings_=grouper.settings,
                )
                solution = solver.solve()
                assert len(solution) <= group.size
                assert solution.shifts.shape[0] == group.shape[0]
                assert solution.result
                assert len(solution) <= solution.result
                assert (
                    sum(array_.pipelines.size for array_ in solution)
                    == group.size
                )
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


def test_solver_example_case():
    settings_ = settings.Settings(
        group_size_limit=4,
        pipeline_size_limit=4,
        common_ops_percent_bound=0.85,
    )
    source_array = np.array(
        [
            [1, 3, 1, 2],
            [1, 1, 1, 2],
            [3, 2, 1, 1],
            [1, 2, 2, 1],
        ],
        dtype=settings_.default_dtype,
    )
    frequency = custom_st.frequency(
        pipelines=source_array,
        settings_=settings_,
    )
    source_matrix = ppao.SourceMatrix(
        from_array=source_array,
        settings_=settings_,
        frequency=frequency,
    )
    assert (source_matrix.most_common == np.array((1, 2))).all()
    assert source_matrix is not None
    solver = ppao.solver.PipelineMatrixSolver(
        source_matrix=source_matrix, settings_=settings_
    )
    solution = solver.solve()
    expected = (
        ExecutionUnit(
            operation=1, pipelines=np.array([0, 1, 1, 3], dtype="uint16")
        ),
        ExecutionUnit(operation=3, pipelines=np.array([0, 2], dtype="uint16")),
        ExecutionUnit(operation=1, pipelines=np.array([0, 1], dtype="uint16")),
        ExecutionUnit(
            operation=2, pipelines=np.array([2, 3, 0, 1, 3], dtype="uint16")
        ),
        ExecutionUnit(
            operation=1, pipelines=np.array([2, 2, 3], dtype="uint16")
        ),
    )
    assert solution.result == 8
    assert (solution.shifts == (-1, -1, 0, 0)).all()

    for key, execution_unit in enumerate(solution):
        if isinstance(
            execution_unit.pipelines == expected[key].pipelines, bool
        ):
            assert execution_unit.pipelines == expected[key].pipelines
        else:
            assert (execution_unit.pipelines == expected[key].pipelines).all()
        assert execution_unit.operation == expected[key].operation
    assert (
        sum(array_.pipelines.size for array_ in solution) == source_matrix.size
    )
