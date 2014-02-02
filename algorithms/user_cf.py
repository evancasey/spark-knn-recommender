""" User-based Collaborative Filtering on pySpark """

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

    ungrouped_users = data.map(
        lambda p : (p[0], (p[1],p[2])))
    user_item_hist = ungrouped_users.groupByKey()
    # item_with_item_hist = user_item_hist.map(
    #     lambda p : (p[0],p[1]))


    # d = item_with_item_hist.collect()

    # print d
    # print [x for x in d]
