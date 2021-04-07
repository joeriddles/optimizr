from __future__ import annotations
from typing import Any, Optional, Tuple

import numpy as np
from operator import itemgetter
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer

from .groups import Group


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


def get_scaled_utility_vector(
    binarized_target_group: csr_matrix,
    extraneous_penalty: float,
    user_access_counts: Optional[Any]=None,
):
    """
    Weights users based on importance (extraneous, matching...)

    param: user_access_counts: ndarray - dense, in same order as user IDs so they line up with binarized version.
    """
    if user_access_counts is None:
        scaled_utility_vector = binarized_target_group.todense().astype('float32')
        scaled_utility_vector[scaled_utility_vector <= 0] =  extraneous_penalty
    else:
        scaled_utility_vector = user_access_counts.astype('float32')
        max_user_access_count = np.max(scaled_utility_vector)
        user_access_counts = user_access_counts / max_user_access_count + 1
        # [... == 1] because we've added 1 to each entry - JR
        scaled_utility_vector[scaled_utility_vector == 1] = extraneous_penalty

    return scaled_utility_vector


def scale_and_score_matrix(
    binarized_groups: csr_matrix,
    binarized_target_group: csr_matrix,
    scaled_utility_vector: np.ndarray, 
    group_weights: Optional[Any]=None,
) -> np.ndarray:
    """
    1. Scales the target group and binary groups
    2. Calculates the __weighted__ distances between target group and every other group
    """

    scaled_target_group = binarized_target_group.multiply(scaled_utility_vector)
    scaled_binarized_groups = binarized_groups.multiply(scaled_utility_vector)
    group_distances: np.ndarray = pairwise_distances(scaled_target_group, scaled_binarized_groups)  # type: ignore
    
    if group_weights is not None:
        # may have to transpose group_weights... - JR
        group_weights = 1 / group_weights
        group_distances = np.multiply(group_distances, group_weights) # type: ignore

    return group_distances


def apply_group_from_index(
    binarized_groups: csr_matrix,
    binarized_target_group: csr_matrix,
    index: int
):
    """
    "I picked this group as one of my matches, remove its users from the group I'm search for" - Clayton
    """
    current_group = binarized_target_group - binarized_groups[index]
    current_group[current_group < 0] = 0
    matches = binarized_target_group.sum() - current_group.sum()
    return current_group, matches


def find_best_n_groups(group_distances: np.ndarray, n: int):
    """
    Finds beginning candidates ("Entry-points / first-steps of paths for greedy")
    """
    group_distances = group_distances[0]
    n_groups = np.argpartition(group_distances, n)
    n_groups = n_groups[:n]
    return n_groups


def find_groups_from_starting_group(
    start_index: int,
    binarized_groups: csr_matrix,
    binarized_target_group: csr_matrix,
    extraneous_penalty: float,
):
    """
    Get a path of groups with a starting-point point of `start_index`

    param: start_index: int - index for starting group
    returns: a list representing a "path" of groups
    """
    selected_group_indices = [start_index]
    x = 1  # maximum number of groups in a result
    current_group, matches = apply_group_from_index(
        binarized_groups, binarized_target_group, start_index)
    while x < 4:
        scaled_utility_vector = get_scaled_utility_vector(current_group, extraneous_penalty)
        group_distances = scale_and_score_matrix(
            binarized_groups,
            current_group,
            scaled_utility_vector
        )
        best_group = find_best_n_groups(group_distances, 1)[0]
        current_group, matches = apply_group_from_index(binarized_groups, current_group, best_group)
        if matches >= 2:  # if current_group adds at least 2 new matching users
            selected_group_indices.append(best_group)
            x += 1
        else:
            x = 5  # change this
    return selected_group_indices


def score_full_path(
    path: list,
    binarized_groups: csr_matrix,
    binarized_target_group: csr_matrix,
    scaled_utility_vector: np.ndarray,
    extraneous_penalty: float,
):
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

    matching_users_count = matching_users.sum()
    extraneous_users_count = extraneous_users.sum()
    unmatched_users_count = unmatched_users.sum()

    matching_percent = matching_users_count / binarized_target_group.sum()
    extraneous_percent = extraneous_users_count / binarized_target_group.sum()

    scaled_matching_users = selected_users.multiply(scaled_utility_vector).sum()
    scaled_unmatched_users = unmatched_users.multiply(scaled_utility_vector).sum() * -1
    score = scaled_matching_users + extraneous_users_count*extraneous_penalty + scaled_unmatched_users

    evaluation_dict = {
        'Score': score,
        'Matching Percent': matching_percent,
        'Matching Users': matching_users.nonzero()[1],
        'Matching Users Count': matching_users_count,
        'Extraneous Users': extraneous_users.nonzero()[1],
        'Extraneous Users Count': extraneous_users_count,
        'Extraneous Percent': extraneous_percent,
        'Unmatched Users': unmatched_users.nonzero()[1],
        'Unmatched Users Count': unmatched_users_count,
        'Selected Groups': path
    }

    return evaluation_dict


def find_optimal_groups(
    binarized_target_group: csr_matrix,
    binarized_groups: csr_matrix,
    n_groups: Optional[int]=None,
    user_access_counts: Optional[Any]=None,
    group_weights: Optional[Any]=None,
):
    if n_groups is None:
        n_groups = 10

    target_group_size = binarized_target_group.sum()
    extraneous_penalty: float = calculate_exraneous_penalty(target_group_size)

    scaled_utility_vector = get_scaled_utility_vector(
        binarized_target_group,
        extraneous_penalty,
        user_access_counts=user_access_counts
    )

    group_distances = scale_and_score_matrix(
        binarized_groups,
        binarized_target_group,
        scaled_utility_vector,
        group_weights=group_weights,
    )

    # AKA: best n starting-points
    best_n_groups = find_best_n_groups(group_distances, n_groups)

    paths: list = []
    for group_index in best_n_groups:
        path = find_groups_from_starting_group(
            group_index,
            binarized_groups,
            binarized_target_group,
            extraneous_penalty
        )
        paths.append(path)

    scores = []
    for path in paths:
        score = score_full_path(
            path,
            binarized_groups,
            binarized_target_group,
            scaled_utility_vector,
            extraneous_penalty,
        )
        scores.append(score)

    sorted_scores = sorted(scores, key=itemgetter('Score'), reverse=True)

    best_choice = sorted_scores[0]

    return best_choice, sorted_scores


def calculate_exraneous_penalty(target_group_size) -> float:
    """Scales amount of extra users allowed using a function of target_group_size"""
    extraneous_penalty = -0.0242654 * (target_group_size ** 0.3389719)
    return extraneous_penalty
