from __future__ import annotations
import os
from pathlib import Path

from . import files
from .GroupOptimizations3 import (
    binarize_groups,
    calculate_exraneous_penalty,
    find_optimal_groups
)


path = Path(__file__).parent.parent


NUM_SEEDS = 10


def main():
    groups = files.read_groups(os.path.join(path, 'data', 'hashed_data.json'))
    target_groups = files.read_groups(os.path.join(path, 'data', 'generated_groups.json'))

    for target_group in target_groups:
        binarized_groups, binarized_target_group, all_members = binarize_groups(groups, target_group)
        target_group_size = binarized_target_group.sum()

        extraneous_penalty: float = calculate_exraneous_penalty(target_group_size)
        best_choice, sorted_scores = find_optimal_groups(
            binarized_target_group,
            binarized_groups,
            extraneous_penalty,
            NUM_SEEDS,
        )

        breakpoint()


if __name__ == '__main__':
    main()
