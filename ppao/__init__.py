"""Python Pipelines Algorithmic Optimizer.

    This module can be useful if you have high-load web application.

    More information: https://github.com/borontov/ppao
"""
from ppao.custom_types import ExecutionUnit, Solution
from ppao.grouper import Grouper
from ppao.matrix import SourceMatrix
from ppao.solver import PipelineMatrixSolver
