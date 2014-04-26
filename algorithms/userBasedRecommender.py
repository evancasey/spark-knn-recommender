# User-based Collaborative Filtering on pySpark with cosine similarity and weighted sums

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

def keyOnFirstUser(user_pair,item_sim_data):
    '''
    For each user-user pair, make the first user's id the key
    '''
    (user1_id,user2_id) = user_pair
    return user1_id,(user2_id,item_sim_data)

def nearestNeighbors(user,users_and_sims,n):
    '''
    Sort the predictions list by similarity and select the top-N neighbors
    '''
    users_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return user, users_and_sims[:n]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1],"PythonUserItemCF")
    lines = sc.textFile(sys.argv[2])

    '''
    Obtain the sparse item-user matrix:
        item_id -> ((user_1,rating),(user2,rating))
    '''
    item_user_pairs = lines.map(parseVectorOnItem).groupByKey()

    '''
    Get all item-item pair combos:
        (user1_id,user2_id) -> [(rating1,rating2),
                                (rating1,rating2),
                                (rating1,rating2),
                                ...]
    '''
    pairwise_users = item_user_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: findUserPairs(p[0],p[1])).groupByKey()

    '''
    Calculate the cosine similarity for each user pair and select the top-N nearest neighbors:
        (user1,user2) ->    (similarity,co_raters_count)
    '''
    user_sims = pairwise_users.map(
        lambda p: calcSim(p[0],p[1])).map(
        lambda p: keyOnFirstUser(p[0],p[1])).groupByKey().map(
        lambda p: nearestNeighbors(p[0],p[1],50)).collect()

    pdb.set_trace()

