# User-Item Collaborative Filtering on pySpark

import sys
from itertools import combinations
import numpy as np

from pyspark import SparkContext


def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    return np.array([float(x) if i == 2 else x for i,x in enumerate(line.split('|'))])

def unpackItemHist(user_id,item_hist):
    '''
    Transpose the into user-item format 
    '''
    for (item_id, rating) in item_hist:
        return item_id, (user_id,item_hist,rating)

def findUserPairs(item_id,user_rating_hist_list):
    ''' 
    For each item, find all user-user pairs 
    '''
    for user1,user2 in combinations(user_rating_hist_list,2):
        return (user1[0],user2[0]),((user1[1],user2[1]),(user1[2],user2[2]),item_id)

def calcSim(user_pair,co_rate_with_hist):
    ''' 
    For each user-user pair, return the specified similarity measure,
    along with co_rated_items_count
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rt in ratings:
        sum_xx += np.float(rt[0]) * np.float(rt[0])
        sum_yy += np.float(rt[1]) * np.float(rt[1])
        sum_xy += np.float(rt[0]) * np.float(rt[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))

    return user_pair,1


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

    ''' 
    Obtain the sparse user-item matrix:
        user_id -> [(item_id_1, rating_1),
                    (item_id_2, rating_2),
                    ...]
    '''
    user_item_pairs = data.map( 
        lambda p : (p[0], (p[1],p[2]))).groupByKey()
    
    '''
    Find all item-user pairs, storing each user's item history
        item_id -> [((user_id_1,[item_hist_1]),rating_1)]
                    ((user_id_2,[item_hist_2]),rating_2),
                    ...]
    '''
    item_user_rating_pairs = user_item_pairs.map(
        lambda p: unpackItemHist(p[0],p[1])).groupByKey()

    '''
    Find all user1-user2 pair combos
        (user1,user2) ->    [(([user1_item_hist],[user2_item_hist]), (co_rating1,co_rating2), corated_item_id_1),
                             (([user1_item_hist],[user2_item_hist]), (co_rating1,co_rating2), corated_item_id_2),
                             ...]
    '''
    pairwise_users = item_user_rating_pairs.map(
        lambda p: findUserPairs(p[0],p[1])).filter(
        lambda p: p is not None).groupByKey()

    
    print pairwise_users