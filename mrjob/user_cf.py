#src: http://aimotion.blogspot.com/2009/11/collaborative-filtering-implementation.html '''

import pdb
from math import sqrt
from optparse import OptionParser
from collections import defaultdict
from utils import fetch_user_hist, write_results

ucf_predict_file = "data/results/user_cf_predict.csv"

def sim_cosine(user_hist, p1, p2):
    ''' Returns a Cosine similarity score for p1 and p2
    '''
    
    si = []
    ind = 0
    for [item1,count1] in user_hist[p1]:
        trunc_user_hist = [r[0] for r in user_hist[p2]]
        if item1 in trunc_user_hist:          
            si.append(ind)
            ind += 1

    # if they have no rating in common, return 0
    if len(si) == 0: 
        return 0

    # sum of the products 
    trunc_count1 = [r[1] for r in user_hist[p1]]
    trunc_count2 = [r[1] for r in user_hist[p2]]
    numerator = sum([trunc_count1[i] * trunc_count2[i] for i in si])

    # sum of squares
    sum1 = sum([trunc_count1[i]**2 for i in si])
    sum2 = sum([trunc_count2[i]**2 for i in si])
    
    # dot product calculation
    denominator = sqrt(sum1) * sqrt(sum2)

    # check for denom == 0
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def sim_pearson(ratings,p1,p2):
    ''' Returns the Pearson correlation coefficient for p1 and p2 
    '''
    # get the list of mutually rated items
    si = {}
    for item in ratings[p1]:
        if item in ratings[p2]: 
            si[item] = 1

    # if they are no rating in common, return 0
    if len(si) == 0:
        return 0

    # sum calculations
    n = len(si)

    # sum of all preferences
    sum1 = sum([ratings[p1][x] for x in si])
    sum2 = sum([ratings[p2][x] for x in si])

    # sum of the squares
    sum1Sq = sum([ratings[p1][x]**2 for x in si])
    sum2Sq = sum([ratings[p2][x]**2 for x in si])

    # sum of the products
    pSum = sum([ratings[p1][x] * ratings[p2][x] for x in si])

    # calculate r (Pearson score)
    num = pSum - (sum1 * sum2/n)
    den = sqrt((sum1Sq - (sum1**2/n)) * (sum2Sq - (sum2**2/n)))
    if den == 0:
        return 0

    r = num/den

    return r

def top_matches(ratings,person,n=5,similarity=sim_cosine):
    ''' Returns the best matches for person from the ratings dictionary
        number of the results and similiraty function are optional params.
    '''
    scores = [(similarity(ratings,person,other),other)
                for other in ratings if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def get_recommendations(user_hist,user,similarity=sim_cosine):
    ''' Gets recommendations for a user by using a weighted average
        of every other user's rankings
    '''
    
    totals = {}
    sim_sums = {}

    for other in user_hist:
        # don't compare user to itself
        if other == user:
            continue

        # call the similarity function we define
        sim = similarity(user_hist,user,other)

        # ignore scores of zero or lower
        if sim <= 0: 
            continue

        # for each of other's items calc totals + sim_sums
        for [item,count] in user_hist[other]:

            trunc_user_hist = [r[0] for r in user_hist[user]]
            
            # only score items user hasn't seen yet
            if item not in trunc_user_hist:
                
                # similarity * score
                totals.setdefault(item,0)
                totals[item] += count * sim
        
                # sum of similarities
                sim_sums.setdefault(item,0)
                sim_sums[item] += sim

    # create the normalized list
    rankings = [(total/sim_sums[item],item) for item,total in totals.items()]

    # return the sorted list
    rankings.sort()
    rankings.reverse()

    # take the top num_recs
    return rankings

def get_batch_recommendations(user_hist,num_recs=500):
    ''' Calls get_recommendations on each user in the user_hist and 
        returns a list of the first n recs for each user
    '''
    
    predicted_results = []
    for user, item_list in user_hist.items():
        score_recs = get_recommendations(user_hist,user)[:num_recs]
        recs = [s[1] for s in score_recs]
        if len(recs) > 0:
            predicted_results.append(recs)

    return predicted_results

if __name__=="__main__":

    parser = OptionParser()
    parser.add_option("-i", "--input", type="string", dest="input_file")  
    parser.add_option("-o", "--output", type="string", dest="output_file")  

    (options,args) = parser.parse_args()

    if options.input_file and options.output_file:
        #read in dataset
        train_data = fetch_user_hist(options.input_file,",")
        
        #call get recommendations on each user
        rec_list = get_batch_recommendations(train_data,500)

        #write out to predict_file
        write_results(rec_list,options.output_file)        
    else:
        raise NameError('Input file not found')