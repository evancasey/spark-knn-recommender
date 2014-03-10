# User-Item Collaborative Filtering on pySpark

import sys
from itertools import combinations
import numpy as np
import pdb

from pyspark import SparkContext


def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def keyOnUser(user_id,item_with_rating):
    '''
    Insert Comment
    '''
    return user_id,item_with_rating

def keyOnItem(user_id,user_rating_with_hist):
    ''' 
    Insert Comment
    '''
    (item_hist,user_with_rating) = user_rating_with_hist
    (item_id,rating) = user_with_rating
    return item_id, (item_hist, user_id, rating)

def findUserPairs(item_id,user_rating_hist_list):
    ''' 
    For each item, find all user-user pairs 
    '''
    user_indices = range(len(user_rating_hist_list))
    user_pairs = []
    for ui1,ui2 in combinations(user_indices,2):
        user_pairs.append(((user_rating_hist_list[ui1][0],user_rating_hist_list[ui2][0]), \
                           (user_rating_hist_list[ui1][1],user_rating_hist_list[ui2][1]), \
                           (user_rating_hist_list[ui1][2],user_rating_hist_list[ui2][2]), item_id))

    return 1,user_pairs

def keyOnUserPairs(rating_pair,user_hists,user_pair,item_id):
    '''
    Insert Comment 
    '''
    return rating_pair,(user_hists,user_pair,item_id)

def combineLists(key,user_pair_rating_hist_list):
    '''
    Insert Comment
    '''
    y = []
    for x in user_pair_rating_hist_list:
        y.append(x)
    return y


def calcSim(user_pair,item_hist_and_rating):
    ''' 
    For each user-user pair, return the specified similarity measure,
    along with co_rated_items_count
    '''

    item_hists,item_co_ratings,co_rated_item_id = item_hist_and_rating
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for u1,u2 in zip(item_hists[0],item_hists[1]):
        sum_xx += np.float(u1) * np.float(u1)
        sum_yy += np.float(u2) * np.float(u2)
        sum_xy += np.float(u1) * np.float(u2)
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))

    return user_pair,(cos_sim,n)


def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1], "PythonUserCF")
    lines = sc.textFile(sys.argv[2])
    data = lines.map(parseVector).cache()


    user_item_hist = data.groupByKey()

    user_item_rating_pairs = data.map(
        lambda p: keyOnUser(p[0],p[1]))  

    '''
    Find all item-user pairs, storing each user's item history
        item_id -> [((user_id_1,[item_hist_1]),rating_1)]
                    ((user_id_2,[item_hist_2]),rating_2),
                    ...]
    '''
    item_with_rating_and_user_hist = user_item_hist.join(user_item_rating_pairs).map(
        lambda p: keyOnItem(p[0],p[1])).groupByKey()

    '''
    Find all user1-user2 pair combos with item_hists, co_ratings, and co_rated_item_id
        (user1,user2) ->    [(([user1_item_hist],[user2_item_hist]), (co_rating1,co_rating2), corated_item_id_1),
                             (([user1_item_hist],[user2_item_hist]), (co_rating1,co_rating2), corated_item_id_2),
                             ...]
    '''
    user_user_pairs = item_with_rating_and_user_hist.map(
        lambda p: findUserPairs(p[0],p[1])).reduceByKey(
        lambda p1,p2: combineLists(p1,p2)).collect()[0][1]

    for p in user_user_pairs:
        print p
    '''
    Get cosine similarity for each user pair
        (item1,item2) ->    (similarity,co_rated_items_count)
    '''

    # user_sims = user_user_pairs.map(
    #     lambda p: calcSim(p[0],p[1][0])).collect()

    # print user_sims