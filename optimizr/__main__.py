from __future__ import annotations
import json

from optimizr.results import Result
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
    all_results: list[list[Result]] = []
    for target_group in target_groups:
        mlb, binarized_groups, all_members = binarize_groups(groups)
        binarized_target_group = binarize_target_group(mlb, target_group)
        best_result, sorted_results = find_optimal_groups(
            binarized_target_group,
            binarized_groups,
        )
        all_results.append(sorted_results)
        print(best_result)

    json_data = [
        [vars(score) for score in scores]
        for scores
        in all_results
    ]
    with open(os.path.join(path, 'data', 'out.json'), mode='w') as outfile:
        json.dump(json_data, outfile)

    def get_members(indices):
        return [
            all_members[index]
            for index
            in indices
        ]

    for results in all_results:
        for result in results:
            result.matching_users = get_members(result.matching_users)
            result.unmatched_users = get_members(result.unmatched_users)
            result.extraneous_users = get_members(result.extraneous_users)

    json_data = [
        [vars(score) for score in scores]
        for scores
        in all_results
    ]
    with open(os.path.join(path, 'data', 'out_real.json'), mode='w') as outfile:
        json.dump(json_data, outfile)


if __name__ == '__main__':
    main()
