import csv
import pdb
from collections import defaultdict
import random

DATA_CF_LOCAL = "data/ratings10m.txt"

if __name__ == "__main__":

    ratings = defaultdict(list)

    # read in the data
    f = open(DATA_CF_LOCAL, 'rt')
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        user = row[0]
        item = row[1]
        rating = row[2]
        ratings[user] += [(item,rating)]

    train = defaultdict(list)
    test = defaultdict(list)

    # break up into train and test 
    for k,v in ratings.items():
        test[k] = random.sample(v,len(v)/10)
        train[k] = [x for x in v if x not in test[k]]

    train_rows = []
    test_rows = []

    # write out to separate txts
    for k,v in train.items():
        for x in v:
            train_rows.append([k,x[0],x[1]])

    for k,v in test.items():
        for x in v:
            test_rows.append([k,x[0],x[1]])
    
    with open("tests/data/cftrain.txt", "w") as text_file:
        a = csv.writer(text_file, delimiter="|")
        a.writerows(train_rows)

    with open("tests/data/cftest.txt", "w") as text_file:
        a = csv.writer(text_file, delimiter="|")
        a.writerows(test_rows)