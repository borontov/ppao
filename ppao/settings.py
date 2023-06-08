from dataclasses import dataclass

from ppao import constants, exceptions


@dataclass(slots=True, frozen=True)
class Settings:
    """Class for define custom ppao settings.

    Attributes:
        common_ops_percent_bound: percent bound of the most common operations.
        common_ops_bound: bound of the most common operations.
        group_size_limit: grouper output size limit.
        pipeline_size_limit: max number of pipeline operations.
        default_dtype: default dtype used by ppao arrays.
        default_shift_array_dtype: default dtype of ppao shift arrays.
        default_array_type_code: default type code of ppao simple arrays.
    """

    common_ops_percent_bound: float = 0.5
    common_ops_bound: int = 3
    group_size_limit: int = 4
    pipeline_size_limit: int = 4
    default_dtype: str = "uint16"
    default_shift_array_dtype: str = "int8"
    default_array_type_code: str = "I"

    def __post_init__(self):
        for k, v in self.__annotations__.items():
            if not isinstance(getattr(self, k), v):
                raise exceptions.TypeValidationError(
                    f"{k} must be of type {v}."
                )
        if self.default_dtype not in constants.ACCEPTABLE_DEFAULT_DTYPE:
            raise exceptions.DefaultDtypeValidationError()

        if self.default_shift_array_dtype not in (
            constants.ACCEPTABLE_DEFAULT_SHIFT_ARRAY_DTYPE
        ):
            raise exceptions.DefaultShiftArrayDtypeValidationError()

        if self.default_array_type_code not in (
            constants.ACCEPTABLE_ARRAY_TYPE_CODE
        ):
            raise exceptions.DefaultArrayTypeCodeValidationError()

        if not 0.01 <= self.common_ops_percent_bound <= 0.99:
            raise exceptions.PercentBoundValidationError()

        if self.common_ops_bound < 1:
            raise exceptions.CommonOpsBoundValidationError()

        if not 2 <= self.group_size_limit <= 32:
            raise exceptions.GroupSizeLimitValidationError()

        if not 2 <= self.pipeline_size_limit <= 5:
            raise exceptions.PipelineSizeLimitValidationError()


DEFAULT_SETTINGS = Settings()
