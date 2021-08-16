# -*- coding: utf-8 -*-
from scipy import sparse
import numpy as np


###############################this is for testing
import random
import string
import time


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
print('started fake data construction')
tic()

def create_random_groups(number_of_users, number_of_groups):
    all_groups = sparse.random(number_of_groups, number_of_users, density = .1, data_rvs=np.ones, format='csr', dtype=np.int8)
    original_group = sparse.random(1, number_of_users, density=.05, data_rvs=np.ones, format='csr', dtype=np.int8)
    print(f'Created random sparse matrix of size ~{all_groups.data.nbytes / 1000000} MB.')
    return all_groups, original_group


#create fictitious groups and list of users losing access, including fake user names, and the importance of each user (access statistics)
number_of_users = 1000
number_of_groups = 5000
all_groups, original_group = create_random_groups(number_of_users, number_of_groups)

fake_user_names = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) for i in range(number_of_users)]
fake_group_names = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)) for i in range(number_of_groups)]

fake_scaled_users_losing_access = original_group.multiply(np.random.randint(low=1, high=3, size=(1,number_of_users)))
toc()
#################################

tic()
print('started calculations')



#magic function to efficiently drop columns of a CSR matrix via temporary conversion to COO matrix format
def dropcols(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()

#aka filter out groups that have <2 matches
def find_relevant_row_search_space(matrix, row):
    #find which groups have users that match the target group
    matched_users_in_row = matrix*row.T
    #eliminate any groups that only have 1 match, useless
    matched_users_in_row[matched_users_in_row < 2] = 0
    #this is also the mapping to get back from the updated index back to the original index
    rows_with_matches = matched_users_in_row.nonzero()[0].tolist()
    #filter all groups matrix to only relevant rows
    relevant_rows = all_groups[rows_with_matches]
    return relevant_rows, rows_with_matches

#remove columns that aren't relevant (either user already chosen or not in search group)
def find_relevant_column_search_space(matrix, row_indices, kind='keep'):
    if kind == 'keep':
        result_matrix = matrix[:,row_indices]
    elif kind == 'drop':
        result_matrix = dropcols(matrix, row_indices)
    return result_matrix
        
def find_cost_per_row(matrix, row_cost_scaler):
    return relevant_rows*row_cost_scaler.T

def find_benefit_per_row(matrix):
    benefit_per_row = matrix.sum(axis=1)
    benefit_per_row[benefit_per_row <2]=-100000 #penalize rows with 0 or 1 matches
    return benefit_per_row


def find_best_group(cost_per_row, relevant_rows_and_columns):
    '''
    This is the big decision-making function that chooses which group to suggest next.
    The "best group" is the one with the highest marginal utility, or cost - benefit.
    For example, a group with 2 matches of benefit 1 and 5 extraneous users with cost 0.1 has a net benefit of 1.5.

    Parameters
    ----------
    cost_per_row : ndarray
        The extraneous user cost associated with selecting each row (group).
    relevant_rows_and_columns : scipy sparse csr matrix
        The columns of users that haven't yet been matched, and rows that can possibly have a matching user.

    Returns
    -------
    relevant_rows_and_columns : scipy sparse csr matrix
        The columns of users that haven't yet been matched, and rows that can possibly have a matching user.

    '''
    matrix_shape = relevant_rows_and_columns.get_shape()
    if matrix_shape[1] > 1:
        #net benefit = matching gain - cost 
        benefit_per_row = find_benefit_per_row(relevant_rows_and_columns)
        net_benefit_per_row = benefit_per_row + cost_per_row
        if np.amax(net_benefit_per_row) < 0:
            print('Warning: no groups found in budget threshold')
        best_group_index = np.argmax(net_benefit_per_row)
        #find which users are matching, and remove those columns in relevant_rows_and_columns, then reappend cost per row, then sum again, then find best, etc
        matching_users_in_selected_group = relevant_rows_and_columns[best_group_index].nonzero()[1].tolist()
        #remove matching users columns
        relevant_rows_and_columns = find_relevant_column_search_space(relevant_rows_and_columns, matching_users_in_selected_group, kind='drop')
        suggested_groups_unmapped.append(best_group_index)
    else:
        print('No users left to match')
    return relevant_rows_and_columns

#function for getting stats on the suggested groups, not needed but helpful for testing 😄
def calculate_suggested_stats(all_groups, users_losing_access_indices, suggested_groups_indices):
    results = []
    all_suggested_users = []
    for suggested_group_index in suggested_groups_indices:
        suggested_users = all_groups[suggested_group_index].nonzero()[1].tolist()
        matched_users = set(users_losing_access_indices) & set(suggested_users)
        matched_users_count = len(matched_users)
        new_matched_users = set(matched_users) - set(all_suggested_users)
        new_matched_users_count = len(new_matched_users)
        extra_users = set(suggested_users) - set(users_losing_access_indices)
        extra_users_count = len(extra_users)
        print(f'On its own group {suggested_group_index} has {matched_users_count} matches ({new_matched_users_count} new) and {extra_users_count} extra')
        for i in suggested_users:
            all_suggested_users.append(i)
    all_matched_users = set(users_losing_access_indices) & set(all_suggested_users)
    all_matched_users_count = len(all_matched_users)
    all_extra_users = set(all_suggested_users) - set(users_losing_access_indices)
    all_extra_users_count = len(all_extra_users)
    unmatched_users = set(users_losing_access_indices) - set(all_suggested_users)
    unmatched_users_count = len(unmatched_users)
    print(f'''\nIn total groups {suggested_groups_indices} have \
          \n\t{all_matched_users_count} matches \
          \n\t{all_extra_users_count} extra \
          \n\t{unmatched_users_count} unmatched''')
    
    return all_matched_users, unmatched_users, all_extra_users

#the indices of the users initially losing access
users_losing_access_indices = original_group.nonzero()[1].tolist()

#this is a positive number representing the "cost" of an extraneous user
extraneous_cost = .1

#a row-scaling vector where matching = 0 and extraneous = extraneous_cost
row_cost_scaler = sparse.csr_matrix(original_group.toarray()-1)*extraneous_cost

#get rid of irrelevant rows that don't have possible matches, but keep a mapping to get back to the original group indexes
relevant_rows, row_mapping = find_relevant_row_search_space(all_groups, original_group)

#scale the relevant rows so some users are more important than others. Make sure they're all >= 1
relevant_rows = relevant_rows.multiply(fake_scaled_users_losing_access).tocsr()

#calculate the "cost" per row based on the extraneous user penalty
cost_per_row = find_cost_per_row(relevant_rows,row_cost_scaler)

#make a smaller array whose columns only represent users from the target group
relevant_rows_and_columns = find_relevant_column_search_space(relevant_rows,users_losing_access_indices, kind='keep')

#the suggested_group indices from the smaller array that need to be mapped back to original array
suggested_groups_unmapped = []

#max number of groups to suggest. If a group has < 2 matches it won't be suggested.
for _ in range(10):
    #iteratively find which columns (users) still need to be matched, and find the group with the highest net benefit (gain - cost, or marginal utility)
    relevant_rows_and_columns = find_best_group(cost_per_row, relevant_rows_and_columns)

#map the indices back to the original large array
suggested_groups_indices = [row_mapping[i] for i in suggested_groups_unmapped]



###### this section is just getting stats and names, not needed but nice 😄
matched_users, unmatched_users, extra_users = calculate_suggested_stats(all_groups, users_losing_access_indices, suggested_groups_indices)

matched_users_names = [fake_user_names[i] for i in matched_users]
unmatched_users_names = [fake_user_names[i] for i in unmatched_users]
extra_users_names = [fake_user_names[i] for i in extra_users]
users_initially_losing_access = [fake_user_names[i] for i in users_losing_access_indices]
suggested_group_names = [fake_group_names[i] for i in suggested_groups_indices]



print('finished calculations')
toc()




'''
#I imagine this will all have to get wrapped up into one function. Here's an example, but you'll have a better idea of what it needs to look like

def find_suggested_groups(all_groups, original_group, scaled_users_losing_access, extraneous_cost, max_number_groups_suggested):
    users_losing_access_indices = original_group.nonzero()[1].tolist()
    row_cost_scaler = sparse.csr_matrix(original_group.toarray()-1)*extraneous_cost
    relevant_rows, row_mapping = find_relevant_row_search_space(all_groups, original_group)
    relevant_rows = relevant_rows.multiply(scaled_users_losing_access).tocsr()
    cost_per_row = find_cost_per_row(relevant_rows,row_cost_scaler)
    relevant_rows_and_columns = find_relevant_column_search_space(relevant_rows,users_losing_access_indices, kind='keep')
    suggested_groups_unmapped = []
    for _ in range(max_number_groups_suggested):
        relevant_rows_and_columns = find_best_group(cost_per_row, relevant_rows_and_columns)
    suggested_groups_indices = [row_mapping[i] for i in suggested_groups_unmapped]
    return suggested_groups_indices

scaled_users_losing_access = fake_scaled_users_losing_access
extraneous_cost = .1
max_number_groups_suggested = 4
suggested_groups_indices = find_suggested_groups(all_groups, original_group, scaled_users_losing_access, extraneous_cost, max_number_groups_suggested)
'''