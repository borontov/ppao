from datetime import timedelta

import numpy as np
import pytest
from hypothesis import Verbosity, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

import ppao
import tests.custom_strategies as custom_st
from ppao import exceptions, settings
from ppao.custom_types import Frequency


@given(
    settings_=custom_st.correct_settings(),
    pipelines=st.data(),
)
@hypothesis_settings(
    verbosity=Verbosity.verbose,
    max_examples=1000,
    deadline=timedelta(seconds=1),
)
def test_incorrect_from_array_type_fail(
    settings_: settings.Settings,
    pipelines,
):
    pipelines = pipelines.draw(custom_st.everything_except(np.ndarray))
    with pytest.raises(exceptions.MatrixAttributeTypeValidationError):
        ppao.SourceMatrix(
            from_array=pipelines,
            frequency=Frequency(total=10, most_common={1, 2, 3}),
        )


@given(
    settings_=custom_st.correct_settings(),
    from_array=custom_st.from_array_incorrect_dtype_random_shape(),
    frequency=custom_st.frequency_random_correct(),
)
@hypothesis_settings(
    verbosity=Verbosity.verbose,
    max_examples=1000,
    deadline=timedelta(seconds=1),
)
def test_incorrect_dtype_or_shape_fail(
    settings_: settings.Settings,
    from_array,
    frequency,
):
    if (
        len(from_array.shape) == 1
        or from_array.shape[1] != settings_.pipeline_size_limit
    ):
        with pytest.raises(exceptions.ArrayShapeError):
            ppao.SourceMatrix(
                from_array=from_array,
                settings_=settings_,
                frequency=frequency,
            )
    else:
        with pytest.raises(exceptions.ArrayDtypeError):
            ppao.SourceMatrix(
                from_array=from_array,
                settings_=settings_,
                frequency=frequency,
            )
