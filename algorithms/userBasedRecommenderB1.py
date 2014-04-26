# User-based Collaborative Filtering on pySpark with cosine similarity and weighted sums using broadcast variables

import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import pdb

from pyspark import SparkContext


def parseVectorOnUser(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Key is user_id, converts each rating to a float.
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def parseVectorOnItem(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Key is item_id, converts each rating to a float.
    '''
    line = line.split("|")
    return line[1],(line[0],float(line[2]))

def findUserPairs(item_id,users_with_rating):
    '''
    For each item, find all user-user pairs combos. (i.e. users with the same item) 
    '''
    for user1,user2 in combinations(users_with_rating,2):
        return (user1[0],user2[0]),(user1[1],user2[1])

def calcSim(user_pair,rating_pairs):
    ''' 
    For each user-user pair, return the specified similarity measure,
    along with co_raters_count.
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return user_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0

def closestNeighbors(user,users_and_sims,n):

    users_and_sims.sort(key=lambda x: x[1][0],reverse=True)

    return user, users_and_sims[:n]

def getAllPredictions(user,closest_neighbors,ui):

    all_predictions = []
    for (neighbor,sim_and_count) in closest_neighbors:
        predictions = [i2 for i2 in ui[neighbor] if i2[0] not in [i1[0] for i1 in ui[user]]]
        all_predictions.append((neighbor,sim_and_count,predictions))

    return user, all_predictions


def getItemHistDiff(user1_with_rating,user2_with_rating):
    '''
    For each user1_with_rating, user2_with_rating pair, emit the set 
    difference of user2's item_hist - user1's item_hist.
    '''
    (user1_id,user1_item_hist) = user1_with_rating
    (user2_id,user2_item_hist) = user2_with_rating
    diff_item_hist = [u2 for u2 in user2_item_hist if u2[0] not in [u1[0] for u1 in user1_item_hist]]
    return (user1_id,user2_id),diff_item_hist

def keyOnFirstUser(user_pair,item_sim_data):
    '''
    For each user-user pair, make the first user's id the key
    '''
    (user1_id,user2_id) = user_pair
    return user1_id,(user2_id,item_sim_data)

def topRecs(user1,items_sim_diff):

   
    # initialize dicts to store the score of each individual item,
    # since an item can exist in more than one item_sim_diff
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for item_sim_diff in items_sim_diff:

        # unpack the data in each item_sim_diff 
        (user2_id,sim_count,item_diff) = item_sim_diff
        (sim,count) = sim_count

        for item_with_rating in item_diff:
            
            # unpack the data in each item_with_rating tuple
            (item_id,rating) = item_with_rating

            # update totals and sim_sums with the rating data
            totals[item_id] += sim * rating
            sim_sums[item_id] += sim

    # create the normalized list of scored items 
    scored_items = [(total/sim_sums[item],item) for item,total in totals.items()]

    # sort the scored items in ascending order
    scored_items.sort()
    scored_items.reverse()

    # take out the item score
    ranked_items = [x[1] for x in scored_items]

    return user1,ranked_items


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1],"PythonUserItemCF")
    lines = sc.textFile(sys.argv[2])

    '''
    Obtain the sparse item-user matrix and filter out items with one or less user interactions:
        item_id -> ((user_1,rating),(user2,rating))
    '''
    item_user_pairs = lines.map(parseVectorOnItem).groupByKey().filter(
        lambda p: len(p[1]) > 1)

    '''
    Get all item-item pair combos:
        (user1_id,user2_id) -> [(rating1,rating2),
                                (rating1,rating2),
                                (rating1,rating2),
                                ...]
    '''
    pairwise_users = item_user_pairs.map(
        lambda p: findUserPairs(p[0],p[1])).groupByKey()

    '''
    Calculate the cosine similarity for each user pair and select the top-N nearest neighbors:
        (user1,user2) ->    (similarity,co_raters_count)
    '''
    user_pair_sims = pairwise_users.map(
        lambda p: calcSim(p[0],p[1])).map(
        lambda p: keyOnFirstUser(p[0],p[1])).groupByKey().map(
        lambda p: closestNeighbors(p[0],p[1],50)).collect()

    pdb.set_trace()

    # # generate a dict of user1 -> [user2, (sim,count)]
    # # us_dict = {}
    # # for (user,data) in user_pair_sims: 
    # #     us_dict[user] = data

    # # us = sc.broadcast(us_dict)

    # ''' 
    # Obtain the the item history for each user, and key
    # on the first user:
    #     user_id -> [(item_id_1, rating_1),
    #                [(item_id_2, rating_2),
    #                 ...]
    # '''
    # user_item = lines.map(parseVectorOnUser)
    # user_item_hist = user_item.groupByKey().collect()

    # # generate a dict of user1 -> [(item,rating)]
    # ui_dict = {}
    # for (user,items) in user_item_hist: 
    #     ui_dict[user] = items

    # ui = sc.broadcast(ui_dict)

    # unrankedPredictions = user_pair_sims.map(
    #     lambda p: getAllPredictions(p[0],p[1],ui.value))


    # # '''
    # # Get the top recs for each user:

    # # '''
    # user_item_recs = unrankedPredictions.map(
    #     lambda p: topRecs(p[0],p[1])).collect()
