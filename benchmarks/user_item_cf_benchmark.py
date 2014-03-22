import pdb
import csv
from math import sqrt
from collections import defaultdict

predicted_results_file = "../data/results/user_cf_predict.csv"

# returns a Cosine similarity score for p1 and p2
def sim_cosine(ratings, p1, p2):

    # get the list of shared_items
    si = []
    songs = []
    ind = 0
    for [song1,count1] in ratings[p1]:
        trunc_ratings = [r[0] for r in ratings[p2]]
        if song1 in trunc_ratings and songs.count(song1) == 0:          
            si.append(ind)
            songs.append(song1)
            ind += 1

    # if they have no rating in common, return 0
    if len(si) == 0: 
        return 0

    try:
        # sum of the products 
        trunc_count1 = [r[1] for r in ratings[p1]]
        trunc_count2 = [r[1] for r in ratings[p2]]
        numerator = sum([trunc_count1[i] * trunc_count2[i] for i in si])
    except IndexError:
        pdb.set_trace()
        
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


# returns the Pearson correlation coefficient for p1 and p2 
def sim_pearson(ratings,p1,p2):
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

# returns the best matches for person from the ratings dictionary
# number of the results and similiraty function are optional params.
def top_matches(ratings,person,n=5,similarity=sim_cosine):
    scores = [(similarity(ratings,person,other),other)
                for other in ratings if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


# gets recommendations for a person by using a weighted average
# of every other user's rankings
def get_recommendations(ratings,person,num_recs,similarity=sim_cosine):
    totals = {}
    sim_sums = {}

    for other in ratings:
        # don't compare user to itself
        if other == person:
            continue

        # call the similarity function we define
        sim = similarity(ratings,person,other)

        # ignore scores of zero or lower
        if sim <= 0: 
            continue

        # for each of other's items calc totals + sim_sums
        for [song,count] in ratings[other]:

            trunc_ratings = [r[0] for r in ratings[person]]
            
            # only score items user hasn't seen yet
            if song not in trunc_ratings:
                
                # similarity * score
                totals.setdefault(song,0)
                totals[song] += count * sim
        
                # sum of similarities
                sim_sums.setdefault(song,0)
                sim_sums[song] += sim

    # create the normalized list
    rankings = [(total/sim_sums[item],item) for item,total in totals.items()]

    # return the sorted list
    rankings.sort()
    rankings.reverse()

    # take the top num_recs
    return rankings[0:num_recs]

def getAllRecommendations(user_song_history_train,num_recs=500):
    predicted_results = []
    for user, song_history in user_song_history_train.items():
        score_recs = get_recommendations(user_song_history_train,user,num_recs)
        recs = [s[1] for s in score_recs]
        if len(recs) > 0:
            predicted_results.append(recs)


    # write predicted to file
    with open(predicted_results_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(predicted_results)

def getData():
    data = defaultdict(list)
    f = open("../data/ratings3.txt",'rt')
    reader = csv.reader(f,delimiter="|")
    for row in reader:
        data[row[0]].append([row[1],float(row[2])])

    return data

if __name__ == "__main__":
    test_data = getData()
    getAllRecommendations(test_data)