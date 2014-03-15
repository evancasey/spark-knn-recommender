# User-Item Collaborative Filtering on pySpark

import sys
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

def keyOnUserPair(item_id,user_and_rating_pair):
    ''' 
    Convert each item and co_rating user pairs to a new vector
    keyed on the user pair ids, with the co_ratings as their value. 
    '''
    (user1_with_rating,user2_with_rating) = user_and_rating_pair
    user1_id,user2_id = user1_with_rating[0],user2_with_rating[0]
    user1_rating,user2_rating = user1_with_rating[1],user2_with_rating[1]
    return (user1_id,user2_id),(user1_rating,user2_rating)

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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1], "PythonUserCF")
    lines = sc.textFile(sys.argv[2])

    ''' 
    Parse each line both ways:
        user_id -> item_id,rating
        item_id -> user_id,rating
    '''
    user_item = lines.map(parseVectorOnUser).cache()
    item_user = lines.map(parseVectorOnItem).cache()

    '''
    Get co_rating users by joining on item_id:
        item_id -> ((user_1,rating),(user2,rating))
    '''
    item_user_pairs = item_user.join(item_user)

    '''
    Key each item_user_pair on the user_pair, then aggregate all rating pairs:
        (user1_id,user2_id) -> [(rating1,rating2),
                                (rating1,rating2),
                                (rating1,rating2),
                                ...]
    '''
    user_item_rating_pairs = item_user_pairs.map(
        lambda p: keyOnUserPair(p[0],p[1])).groupByKey()

    '''
    Get cosine similarity for each user pair
        (user1,user2) ->    (similarity,co_raters_count)
    '''
    user_pair_sims = user_item_rating_pairs.map(
        lambda p: calcSim(p[0],p[1])).collect()

    for p in user_pair_sims:
        print p