""" Item-based Collaborative Filtering on pySpark """

import sys

import numpy as np
from pyspark import SparkContext


def parseVector(line):
    return np.array([float(x) if i != 1 else x for i,x in enumerate(line.split('|'))])

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

    user_id -> [(item_id, rating),...]
    '''

    user_item_rating_pairs = data.map(
        lambda p : (p[0], (p[1],p[2]))).groupByKey()