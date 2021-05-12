from __future__ import annotations
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer

from .groups import Group
from .results import Result


def binarize_groups(groups: list[Group]) -> Tuple[MultiLabelBinarizer, csr_matrix, np.ndarray]:
    group_members = [
        group.members
        for group
        in groups
    ]

    mlb = MultiLabelBinarizer(sparse_output=True) # type: ignore
    binarized_groups: csr_matrix = mlb.fit_transform(group_members) # type: ignore

    # get all unique members from the transformer
    all_members = mlb.classes_
    return mlb, binarized_groups, all_members


def binarize_target_group(mlb: MultiLabelBinarizer, target_group: Group) -> csr_matrix:
    target_group_members = target_group.members
    y = [target_group_members]
    binarized_target_group: csr_matrix = mlb.transform(y) # type: ignore
    return binarized_target_group


def get_scaled_target_group(
    binarized_target_group: csr_matrix,
) -> np.ndarray:
    scaled_target_group = np.array(binarized_target_group.todense().astype('float32'))
    return scaled_target_group


def scale_and_score_matrix(
    binarized_groups: csr_matrix,
    binarized_target_group: csr_matrix,
    scaled_target_group: np.ndarray, 
) -> np.ndarray:
    scaled_target_group = binarized_target_group.multiply(scaled_target_group) # type: ignore
    scaled_binarized_groups = binarized_groups.multiply(scaled_target_group)
    group_distances: np.ndarray = pairwise_distances(scaled_target_group, scaled_binarized_groups)  # type: ignore
    return group_distances


def apply_group_from_index(
    binarized_groups: csr_matrix,
    binarized_target_group: csr_matrix,
    index: np.number,
):
    """
    "I picked this group as one of my matches, remove its users from the group I'm search for" - Clayton
    """
    current_group = binarized_target_group - binarized_groups[index]
    current_group[current_group < 0] = 0
    matches = len(binarized_target_group.nonzero()[0]) - len(current_group.nonzero()[0])
    return current_group, matches


def find_best_starting_groups(group_distances: np.ndarray, n: int) -> np.ndarray:
    """
    Finds beginning candidates ("Entry-points / first-steps of paths for greedy")
    """
    group_distances = group_distances[0]
    n_groups: np.ndarray = np.argpartition(group_distances, n)
    n_groups = n_groups[:n]
    return n_groups


def find_groups_from_starting_group(
    start_index: np.number,
    binarized_groups: csr_matrix,
    binarized_target_group: csr_matrix,
) -> list[np.number]:
    """
    Get a path of groups with a starting-point point of `start_index`

    param: start_index: int - index for starting group
    returns: a list representing a "path" of groups
    """
    selected_group_indices: list[np.number] = [start_index]
    x = 1  # maximum number of groups in a result
    current_group, matches = apply_group_from_index(binarized_groups, binarized_target_group, start_index)

    member_count = len(binarized_target_group.nonzero()[0])
    if member_count <= 6:
        min_matches = 1
    else:
        min_matches = 2

    while matches >= 2 and x < 4:
        scaled_target_group = get_scaled_target_group(current_group)
        group_distances = scale_and_score_matrix(
            binarized_groups,
            current_group,
            scaled_target_group
        )
        best_group: np.number = find_best_starting_groups(group_distances, 1)[0]
        current_group, matches = apply_group_from_index(binarized_groups, current_group, best_group)
        if matches >= min_matches: # if current_group adds at least 2 new matching users
            selected_group_indices.append(best_group)
            x += 1

    return selected_group_indices


def score_full_path(
    path: list[np.number],
    binarized_groups: csr_matrix,
    binarized_target_group: csr_matrix,
    scaled_target_group: np.ndarray,
) -> Result:
    """
    Calculate final score and other statistics for a path.
    """
    first_group = path[0]
    selected_users = binarized_groups[first_group]
    for group in path:
        selected_users += binarized_groups[group]
    selected_users[selected_users > 1] = 1

    matching_users = selected_users.multiply(binarized_target_group)
    extraneous_users = selected_users - matching_users
    unmatched_users = binarized_target_group - matching_users

    matching_users_count = len(matching_users.nonzero()[0])
    extraneous_users_count = len(extraneous_users.nonzero()[0])
    unmatched_users_count = len(unmatched_users.nonzero()[0])

    matching_percent = matching_users_count / len(binarized_target_group.nonzero()[0])
    extraneous_percent = extraneous_users_count / len(binarized_target_group.nonzero()[0])

    scaled_matching_users_score = selected_users.multiply(scaled_target_group).sum()
    scaled_extraneous_users_score = extraneous_users.multiply(scaled_target_group).sum()
    scaled_unmatched_users_score = unmatched_users.multiply(scaled_target_group).sum() * -1
    score = scaled_matching_users_score + scaled_extraneous_users_score + scaled_unmatched_users_score

    suggested_groups = [
        int(index)
        for index
        in path
    ]

    score = float(score)
    matching_users = matching_users.nonzero()[1].tolist()
    extraneous_users = extraneous_users.nonzero()[1].tolist()
    unmatched_users = unmatched_users.nonzero()[1].tolist()
    matching_users_count = int(matching_users_count)
    extraneous_users_count = int(extraneous_users_count)
    unmatched_users_count = int(unmatched_users_count)
    matching_percent = float(matching_percent)
    extraneous_percent = float(extraneous_percent)
    
    result = Result(
        score,
        suggested_groups,
        matching_percent,
        matching_users,
        matching_users_count,
        extraneous_percent,
        extraneous_users,
        extraneous_users_count,
        unmatched_users,
        unmatched_users_count,
    )

    return result


def find_optimal_groups(
    binarized_target_group: csr_matrix,
    binarized_groups: csr_matrix,
    n_groups: Optional[int]=None,
    weight_members: Optional[Callable[[np.ndarray], np.ndarray]]=None,
    weight_results: Optional[Callable[[Result], Result]]=None,
) -> list[Result]:
    if n_groups is None:
        n_groups = 10

    target_group = get_scaled_target_group(binarized_target_group)

    # Weight members
    if weight_members is not None:
        target_group = weight_members(target_group)

    # Scale and score
    group_distances = scale_and_score_matrix(
        binarized_groups,
        binarized_target_group,
        target_group,
    )

    best_starting_groups = find_best_starting_groups(group_distances, n_groups)

    paths: list = []
    for group_index in best_starting_groups:
        path: list[np.number] = find_groups_from_starting_group(
            group_index,
            binarized_groups,
            binarized_target_group,
        )
        paths.append(path)

    # Score combinations of groups
    results: list[Result] = []
    for path in paths:
        result = score_full_path(
            path,
            binarized_groups,
            binarized_target_group,
            target_group,
        )
        results.append(result)

    # Shift scores right
    min_score = min([result.score for result in results])
    for result in results:
        result.score += min_score

    # # Weight groups
    if weight_results is not None:
        for index, result in enumerate(results):
            weighted_result = weight_results(result)
            results[index] = weighted_result

    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    return sorted_results


def calculate_exraneous_penalty(target_group_size) -> float:
    """Scales amount of extra users allowed using a function of target_group_size"""
    extraneous_penalty = -0.4 * (target_group_size ** 0.3389719)
    return extraneous_penalty
