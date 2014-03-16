# Item-Item Collaborative Filtering on pySpark 

import sys
from itertools import combinations
import numpy as np

from pyspark import SparkContext


def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def ratingsSumCount(user_id,items_with_rating):
    '''
    Aggregate the item rating history of each user, counting number of 
    ratings and the sum of their ratings
    '''
    item_count = 0.0
    item_sum = 0.0
    item_hist = []
    for item_with_rating in items_with_rating:
        item_count += 1.0
        item_sum += item_with_rating[1]

    return user_id,(item_count,item_sum,items_with_rating)

def findItemPairs(user_id,items_with_rating):
    '''
    For each user, find all item-item pairs combos. (i.e. items with the same user) 
    '''
    for item1,item2 in combinations(items_with_rating,2):
        return (item1[0],item2[0]),(item1[1],item2[1])

def calcSim(item_pair,rating_pairs):
    ''' 
    For each item-item pair, return the specified similarity measure,
    along with co_raters_count
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

    return item_pair, (cos_sim,n)


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
    Obtain the sparse user-item matrix
        user_id -> [(item_id_1, rating_1),
                   [(item_id_2, rating_2),
                    ...]
    '''
    user_item_pairs = data.groupByKey()

    '''
    Get all item-item pair combos
        (item1,item2) ->    [(item1_rating,item2_rating),
                             (item1_rating,item2_rating),
                             ...]
    '''

    item_item_pairs = user_item_pairs.map(
        lambda p: findItemPairs(p[0],p[1])).groupByKey()

    '''
    Calculate the cosine similarity for each item pair
        (item1,item2) ->    (similarity,co_raters_count)
    '''

    item_sims = item_item_pairs.map(
        lambda p: calcSim(p[0],p[1])).collect()

    print item_sims
