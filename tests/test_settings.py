import pytest
from hypothesis import Verbosity, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from ppao import constants, exceptions, settings
from tests.custom_strategies import everything_except


@given(
    default_dtype=st.sampled_from(constants.ACCEPTABLE_DEFAULT_DTYPE),
    common_ops_percent_bound=st.floats(min_value=0.01, max_value=0.99),
    common_ops_bound=st.integers(min_value=1),
    group_size_limit=st.integers(min_value=2, max_value=7),
    pipeline_size_limit=st.integers(min_value=2, max_value=5),
    default_array_type_code=st.sampled_from(
        constants.ACCEPTABLE_ARRAY_TYPE_CODE
    ),
    default_shift_array_dtype=st.sampled_from(
        constants.ACCEPTABLE_DEFAULT_SHIFT_ARRAY_DTYPE
    ),
)
@hypothesis_settings(verbosity=Verbosity.verbose, max_examples=500)
def test_settings_validation_success(
    default_dtype,
    common_ops_percent_bound,
    common_ops_bound,
    group_size_limit,
    pipeline_size_limit,
    default_array_type_code,
    default_shift_array_dtype,
):
    settings.Settings(
        default_dtype=default_dtype,
        common_ops_percent_bound=common_ops_percent_bound,
        common_ops_bound=common_ops_bound,
        group_size_limit=group_size_limit,
        pipeline_size_limit=pipeline_size_limit,
        default_array_type_code=default_array_type_code,
        default_shift_array_dtype=default_shift_array_dtype,
    )


@given(
    default_dtype=st.text().filter(
        lambda default_dtype: default_dtype
        not in constants.ACCEPTABLE_DEFAULT_DTYPE
    ),
    common_ops_percent_bound=st.floats().filter(
        lambda common_ops_percent_bound: 0.99 < common_ops_percent_bound
        or common_ops_percent_bound < 0.01
    ),
    common_ops_bound=st.integers().filter(
        lambda common_ops_bound: 1 > common_ops_bound
    ),
    group_size_limit=st.integers().filter(
        lambda group_size_limit: 2 > group_size_limit or group_size_limit > 7
    ),
    pipeline_size_limit=st.integers().filter(
        lambda pipeline_size_limit: 2 > pipeline_size_limit
        or 5 < pipeline_size_limit
    ),
    default_array_type_code=st.text().filter(
        lambda default_array_type_code: default_array_type_code
        not in constants.ACCEPTABLE_ARRAY_TYPE_CODE
    ),
    default_shift_array_dtype=st.text().filter(
        lambda default_shift_array_dtype: default_shift_array_dtype
        not in constants.ACCEPTABLE_DEFAULT_SHIFT_ARRAY_DTYPE
    ),
)
@hypothesis_settings(verbosity=Verbosity.verbose, max_examples=500)
def test_settings_validation_fail(
    default_dtype,
    common_ops_percent_bound,
    common_ops_bound,
    group_size_limit,
    pipeline_size_limit,
    default_array_type_code,
    default_shift_array_dtype,
):
    with pytest.raises(exceptions.SettingValidationError):
        settings.Settings(
            default_dtype=default_dtype,
            common_ops_percent_bound=common_ops_percent_bound,
            common_ops_bound=common_ops_bound,
            group_size_limit=group_size_limit,
            pipeline_size_limit=pipeline_size_limit,
            default_array_type_code=default_array_type_code,
            default_shift_array_dtype=default_shift_array_dtype,
        )


@given(
    default_dtype=everything_except(str),
    common_ops_percent_bound=everything_except(float),
    common_ops_bound=everything_except(int),
    group_size_limit=everything_except(int),
    pipeline_size_limit=everything_except(int),
    default_array_type_code=everything_except(str),
    default_shift_array_dtype=everything_except(str),
)
@hypothesis_settings(verbosity=Verbosity.verbose, max_examples=50)
def test_settings_validation_bad_types_fail(
    default_dtype,
    common_ops_percent_bound,
    common_ops_bound,
    group_size_limit,
    pipeline_size_limit,
    default_array_type_code,
    default_shift_array_dtype,
):
    with pytest.raises(exceptions.TypeValidationError):
        settings.Settings(
            default_dtype=default_dtype,
            common_ops_percent_bound=common_ops_percent_bound,
            common_ops_bound=common_ops_bound,
            group_size_limit=group_size_limit,
            pipeline_size_limit=pipeline_size_limit,
            default_array_type_code=default_array_type_code,
            default_shift_array_dtype=default_shift_array_dtype,
        )
