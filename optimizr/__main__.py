from __future__ import annotations
import os
from pathlib import Path

from . import files
from .optimizr import (
    binarize_groups,
    binarize_target_group,
    find_optimal_groups
)


path = Path(__file__).parent.parent


def main():
    groups = files.read_groups(os.path.join(path, 'data', 'hashed_data.json'))
    target_groups = files.read_groups(os.path.join(path, 'data', 'generated_groups.json'))
    for target_group in target_groups:
        mlb, binarized_groups, all_members = binarize_groups(groups)
        binarized_target_group = binarize_target_group(mlb, target_group)
        best_choice, sorted_scores = find_optimal_groups(
            binarized_target_group,
            binarized_groups,
        )
        print(best_choice)


if __name__ == '__main__':
    main()
