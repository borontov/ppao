"""Expected ppao exceptions."""
import numpy as np


class SettingValidationError(Exception):
    """The base exception for setting validation errors.

    Attributes:
        msg_prefix: a prefix of exception messages.
    """

    msg_prefix: str = "Setting validation error: "


class PercentBoundValidationError(SettingValidationError):
    """Raises if common_ops_percent_bound does not match constraints."""

    def __init__(
        self,
        msg: str = "common_ops_percent_bound must obey the "
        "condition: 0.01 > common_ops_percent_bound > 0.99.",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class CommonOpsBoundValidationError(SettingValidationError):
    """Error that occurs when common_ops_bound does not match constraints."""

    def __init__(
        self,
        msg: str = "common_ops_bound must obey this condition: "
        "1 <= common_ops_bound",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class GroupSizeLimitValidationError(SettingValidationError):
    """Error that occurs when group_size_limit does not match constraints."""

    def __init__(
        self,
        msg: str = "group_size_limit must obey this condition: "
        "2 <= group_size_limit <= 32",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class PipelineSizeLimitValidationError(SettingValidationError):
    """Raises if pipeline_size_limit does not match constraints."""

    def __init__(
        self,
        msg: str = "pipeline_size_limit must obey this condition: "
        "2 <= pipeline_size_limit <= 5",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class TypeValidationError(SettingValidationError):
    """Raises when the setting type does not match the annotation."""

    def __init__(self, msg: str, *args) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class DefaultDtypeValidationError(SettingValidationError):
    """Error that occurs when the default_dtype is not an unsigned integer."""

    def __init__(
        self, msg: str = "default_dtype must be an unsigned integer", *args
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class DefaultShiftArrayDtypeValidationError(SettingValidationError):
    """Error that occurs when the default_shift_array_dtype is not
    represented in constants.ACCEPTABLE_DEFAULT_SHIFT_ARRAY_DTYPE"""

    def __init__(
        self,
        msg: str = "default_shift_array_dtype must be in"
        " constants.ACCEPTABLE_DEFAULT_SHIFT_ARRAY_DTYPE",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class DefaultArrayTypeCodeValidationError(SettingValidationError):
    """Error that occurs when the default_array_type_code is
    not represented in constants.ACCEPTABLE_ARRAY_TYPE_CODE."""

    def __init__(
        self,
        msg: str = "default_array_type_code must be in"
        " constants.ACCEPTABLE_ARRAY_TYPE_CODE",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class GrouperError(Exception):
    """The base exception for Grouper errors.

    Attributes:
        msg_prefix: a prefix of exception messages.
    """

    msg_prefix: str = "Grouper error: "


class PipelinesShapeError(GrouperError):
    """Error that occurs when a shape of an array is incorrect."""

    def __init__(
        self,
        length: int,
        msg: str = "added pipelines must be of length ",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg + str(length), *args)


class ArraysConcatError(GrouperError):
    """Error that occurs when array concatenation can't be performed."""

    def __init__(
        self,
        array_1: np.ndarray,
        array_2: np.ndarray,
        msg: str = "Can't concatenate the arrays: ",
        *args,
    ) -> None:
        super().__init__(
            super().msg_prefix + msg + str(array_1) + str(array_2), *args
        )


class CreateArrayError(GrouperError):
    """Error that occurs when an array can't be created from input."""

    def __init__(
        self, msg: str = "Can't create array from input.", *args
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class IndexValidationError(Exception):
    """A data does not correspond to the properties of the indexes."""

    def __init__(
        self, msg: str = "Any indexes must be of type int (positive)", *args
    ) -> None:
        super().__init__(msg, *args)


class MatrixError(Exception):
    """The base exception for matrix errors.


    Attributes:
        msg_prefix: a prefix of exception messages.
    """

    msg_prefix: str = "Matrix error: "


class ArrayShapeError(MatrixError):
    """Error that occurs when a shape of a matrix array is incorrect."""

    def __init__(
        self, shape: tuple, msg: str = "added array must be of shape: ", *args
    ) -> None:
        super().__init__(super().msg_prefix + msg + str(shape), *args)


class MatrixAttributeTypeValidationError(MatrixError):
    """Error that occurs when a type of any matrix attribute is incorrect."""

    def __init__(
        self, msg: str = "an attribute has incorrect type.", *args
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class ArrayDtypeError(MatrixError):
    """Error that occurs when a dtype of a matrix array is incorrect."""

    def __init__(
        self,
        dtype: np.dtype,
        msg: str = "added array must be of dtype: ",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg + str(dtype), *args)


class PipelineMatrixSolverError(Exception):
    """The base exception for PipelineMatrixSolver errors.


    Attributes:
        msg_prefix: a prefix of exception messages.
    """

    msg_prefix: str = "PipelineMatrixSolver error: "


class MostCommonIsEmptyError(PipelineMatrixSolverError):
    """The self.source_matrix.most_common variable is empty."""

    def __init__(
        self,
        msg: str = "self.source_matrix.most_common is empty, "
        "try to change common_ops_percent_bound setting.",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class CustomTypeValidationError(Exception):
    """The base exception for custom type validation errors.

    Attributes:
        msg_prefix: a prefix of exception messages.
    """

    msg_prefix: str = "Custom type validation error: "


class CustomTypeAttributeTypeValidationError(CustomTypeValidationError):
    """Error that occurs when a type of custom type attribute is incorrect."""

    def __init__(
        self,
        msg: str = "An attribute of custom type has an incorrect type.",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)


class CustomTypeEmptyArrayError(CustomTypeValidationError):
    """Error that occurs when a custom type array attribute is empty."""

    def __init__(
        self,
        msg: str = "array is empty.",
        *args,
    ) -> None:
        super().__init__(super().msg_prefix + msg, *args)
