from .optimizr import (
    binarize_groups,
    binarize_target_group,
    find_optimal_groups
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
