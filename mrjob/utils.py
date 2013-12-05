#helper functions: data transformations, and csv read/write operations

from collections import defaultdict
import csv
import pdb
import random

def fetch_user_hist(file,sep='|'):
    ''' Reads in specified data file into a user history dict
    '''

    data = defaultdict(list)
    f = open(file, 'rt')
    reader = csv.reader(f,delimiter=sep)
    for row in reader:
        user = row[0]
        item = row[1]
        count = float(row[2])
        data[user].append([item, count])

    return data

def truncate_dict(user_hist_dict, num):
    ''' Truncate user_hist_dict to a subset of num entries 
    '''

    ufkeys = user_hist_dict.keys()[:num]
    user_hist_dict_subset = {}
    for key in ufkeys:
        user_hist_dict_subset[key] = user_hist_dict[key]

    return user_hist_dict_subset

def write_train_and_test_data(user_hist_dict, train_file, test_file):
    ''' Randomly splits each user history item into train and test 
        partitions and then writes out to csv 
    '''

    user_hist_test = {}
    for user,item_list in user_hist_dict.items():
        user_hist_test[user] = random.sample(item_list, len(item_list)/2)

    user_hist_train = {}
    for user,item_list in user_hist_dict.items():
        user_hist_train[user] = [item for item in user_hist_dict[user] if item not in user_hist_test[user]]
    
    with open(test_file, "wb") as f:
        writer = csv.writer(f)        
        writer.writerows(convert_dict_to_list(user_hist_test))

    with open(train_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(convert_dict_to_list(user_hist_train,True))

def convert_dict_to_list(dict,train=False):
    ''' Converts a dict to a 1 dimensional list of [key,value] pairs 
    '''
    
    dict_list = []
    for key, value in dict.iteritems():           
        if train:    
            for v in value:                        
                dict_list.append([key.replace(" ",""),v[0].replace(",","").replace(" ",""),v[1]])
        else:                        
            dict_list.append([item[0].replace(",","").replace(" ","") for item in value])
        
    return dict_list

def write_results(rec_list, output_file):
    ''' Writes out results from a list to output_file '''

    with open(output_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(rec_list)