import json

import numpy as np
from operator import itemgetter
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer


def get_groups():
    mlb = MultiLabelBinarizer(sparse_output=True)  # type: ignore

    # get all the groups to choose from
    with open('hashed_data.json') as groups_json:
        groups_data = json.load(groups_json)
        groups_members = [x['Members'] for x in groups_data]
        groups_names = [x['Group'] for x in groups_data]

    # fit our transformer, and transform groups to choose from
    binarized_groups = mlb.fit_transform(groups_members)

    # get target groups to try to match
    with open('generated_groups.json') as target_groups_json:
        target_data = json.load(target_groups_json)
        target_groups = [x['members'] for x in target_data]

    # transform groups to try to match
    binarized_target_groups = mlb.transform(target_groups)

    # get users from the transformer
    users = mlb.classes_
    return binarized_groups, binarized_target_groups, groups_names, users


# eventually adding user_importances here - CG
def get_scaled_utility_vector(binarized_target_group, extraneous_penalty, user_access_counts):
    """
    Weights users based on importance (extraneous, matching...)

    param: user_access_counts: ndarray - dense, in same order as user IDs so they line up with binarized version.
    """
    scaled_utility_vector = user_access_counts.astype('float32')
    max_user_access_count = np.max(scaled_utility_vector)
    user_access_counts = user_access_counts / max_user_access_count + 1
    # [... == 1] because we've added 1 to each entry - JR
    scaled_utility_vector[scaled_utility_vector == 1] = extraneous_penalty
    return scaled_utility_vector


# eventually adding group_importances here - CG
def scale_and_score_matrix(binarized_groups, binarized_target_group, scaled_utility_vector, group_weights):
    """
    1. Scales the target group and binary groups
    2. Calculates the __weighted__ distances between target group and every other group
    """
    scaled_target_group = binarized_target_group.multiply(scaled_utility_vector)
    scaled_binarized_groups = binarized_groups.multiply(scaled_utility_vector)
    group_distances = pairwise_distances(
        scaled_target_group, scaled_binarized_groups)  # type: ignore
    # may have to transpose group_weights... - JR
    group_weights = 1 / group_weights
    group_distances = np.multiply(group_distances, group_weights)
    return group_distances


def apply_group_from_index(binarized_groups, binarized_target_group, index):
    """
    "I picked this group as one of my matches, remove its users from the group I'm search for" - Clayton
    """
    current_group = binarized_target_group - binarized_groups[index]
    current_group[current_group < 0] = 0
    matches = binarized_target_group.sum() - current_group.sum()
    return current_group, matches


def find_best_n_groups(group_distances, n):
    """
    Finds beginning candidates ("Entry-points / first-steps of paths for greedy")
    """
    group_distances = group_distances[0]
    n_groups = np.argpartition(group_distances, n)
    n_groups = n_groups[:n]
    return n_groups


def find_groups_from_starting_group(start_index, binarized_groups, binarized_target_group, extraneous_penalty):
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
            binarized_groups, current_group, scaled_utility_vector)
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
    unmatched_penalty: float,
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


def do_everything(binarized_target_group, binarized_groups, extraneous_penalty, n_groups):
    scaled_utility_vector = get_scaled_utility_vector(binarized_target_group, extraneous_penalty)

    group_distances = scale_and_score_matrix(
        binarized_groups, binarized_target_group, scaled_utility_vector)

    # AKA: best n starting-points
    best_n_groups = find_best_n_groups(group_distances, n_groups)

    paths = []
    for group_index in best_n_groups:
        path = find_groups_from_starting_group(
            group_index, binarized_groups, binarized_target_group, extraneous_penalty)
        paths.append(path)

    scores = []
    for path in paths:
        score = score_full_path(path, binarized_groups, binarized_target_group,
                                scaled_utility_vector, extraneous_penalty, unmatched_penalty)
        scores.append(score)

    sorted_scores = sorted(scores, key=itemgetter('Score'), reverse=True)

    best_choice = sorted_scores[0]

    return best_choice, sorted_scores

# lower values mean more matches but more extraneous. Very sensitive to changes, suggested range (-.04 : -.15 )


#extraneous_penalty = -.12
unmatched_penalty = -1

# pulling in groups from given files; this will be different for you. You'll also probably only pass in one target group
binarized_groups, binarized_target_groups, groups_names, users = get_groups()


# you'll need to provide your target group and all groups
binarized_target_group = binarized_target_groups[3]
target_group_size = binarized_target_group.sum()

# Scales amount of extra users allowed using a function of group_size
extraneous_penalty = -0.0242654*(target_group_size**0.3389719)

n_groups = 10
best_result, best_n_results = do_everything(
    binarized_target_group, binarized_groups, extraneous_penalty, n_groups)


# this stuff was just for generating data for comparisons
'''
matching_percents = []
extraneous_percents = []

for binarized_target_group in binarized_target_groups:
    target_group_size = binarized_target_group.sum()
    extraneous_penalty = -0.0242654*target_group_size**0.3389719
    n_groups = 10
    best_result, best_n_results = do_everything(binarized_target_group, binarized_groups, extraneous_penalty, n_groups)
    matching_percents.append(best_result['Matching Percent'])
    extraneous_percents.append(best_result['Extraneous Percent'])
    print('did one')

with open('hashed_generated_results.json') as groups_result_json:
    result_data = json.load(groups_result_json)
    
joe_matching_percent = [result_data[group]['matching_percentage'] for group in result_data]
joe_extra_percent = [result_data[group]['extra_percentage'] for group in result_data]
df = pd.DataFrame()
df['Original Matching Percentage'] = joe_matching_percent
df['Original Extra Percentage'] = joe_extra_percent
df['New Matching Percentage'] = matching_percents
df['New Extra Percentage'] = extraneous_percents

df.to_csv('results.csv')
'''

'''
matching_percentages = [result_data[x]['matching_percentage'] for x in result_data]
df = pd.DataFrame(matching_percentages, columns = ['Matching Percent'])
df.plot.hist(bins=50)
plt.xlabel('Matching Percent')

'''
