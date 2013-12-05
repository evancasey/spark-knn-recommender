#script to train and test cf algos 

import pdb
import subprocess as sp
import sys
import os
from map import kdd_mapk
from utils import fetch_user_hist, truncate_dict, write_train_and_test_data

sample_data_file = "data/sampling.csv"
movie_ratings_file = "data/ratings.csv"
test_file = "data/results/user_cf_test.csv"
train_file = "data/results/user_cf_train.csv"
ucf_predict_file = "data/results/user_cf_predict.csv"
ucfmr_predict_file = "data/results/user_cf_mr_predict.csv"

if __name__ == '__main__':

    #read in entire dataset
    user_hist_dict = fetch_user_hist(sample_data_file)

    #truncate, if necessary
    user_hist_dict_subset = truncate_dict(user_hist_dict, 1000)
    
    #take random sample of train and test obs and write to .csv's
    write_train_and_test_data(user_hist_dict_subset, train_file, test_file)
    
    #run user_cf and have it output values to ucf_predict_file
    cmd = ["python", "mrjob/user_cf.py", "-i", train_file, "-o", ucf_predict_file]
    p = sp.call(cmd)
    
    #run user_cf_mr and have it output values to ucf_mr_predict_file
    os.system("python mrjob/user_cf_mr.py data/results/user_cf_train.csv > data/results/user_cf_mr_predict.csv")

    print "user_cf_mr: ", kdd_mapk(test_file,ucfmr_predict_file,500)

    print "user_cf: ", kdd_mapk(test_file,ucf_predict_file,500)