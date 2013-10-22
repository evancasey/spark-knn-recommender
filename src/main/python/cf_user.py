''' Implementation of user-based collaborative filtering with nearest neighbor
	and Pearson correlation coefficient similarity measures. Based off of Marcel
	Caraciolo's CF code: http://aimotion.blogspot.com/2009/11/collaborative-filtering-implementation.html '''

# sample dataset
movies={'Marcel Caraciolo': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 
 'The Night Listener': 3.0},
'Luciana Nunes': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 
 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, 
 'You, Me and Dupree': 3.5}, 
'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
 'The Night Listener': 4.5, 'Superman Returns': 4.0, 
 'You, Me and Dupree': 2.5},
'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 
 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0}, 
'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}

import pdb
from math import sqrt

def load_dataset(path=""):
	""" To load the dataSet"
		Parameter: The folder where the data files are stored
		Return: the dictionary with the data
	"""
	#Recover the titles of the books
	books = {}
	for line in open(path+"BX-Books.csv"):
		line = line.replace('"', "")
		(id,title) = line.split(";") [0:2]
		books[id] = title
	
	#Load the data
	prefs = {}
	count = 0
	for line in open(path+"BX-Book-Ratings.csv"):
		line = line.replace('"', "")
		line = line.replace("\\","")
		(user,bookid,rating) = line.split(";")
		try:
			if float(rating) > 0.0:
				prefs.setdefault(user,{})
				prefs[user][books[bookid]] = float(rating)
		except ValueError:
			count+=1
			print "value error found! " + user + bookid + rating
		except KeyError:
			count +=1
			print "key error found! " + user + " " + bookid
	return prefs

# returns a distance-base similarity score for person1 and person2
def sim_distance(prefs, person1, person2):
	# get the list of shared_items
	si = {}
	for item in prefs[person1]:
		if item in prefs[person2]:
			si[item] = 1

	# if they have no rating in common, return 0
	if len(si) == 0: 
		return 0

	# add up the squares of all differences
	sum_of_squares = sum([pow(prefs[person1][item]-prefs[person2][item],2) for item in prefs[person1] if item in prefs[person2]])

	return 1 / (1 + sum_of_squares)


# returns the Pearson correlation coefficient for p1 and p2 
def sim_pearson(prefs,p1,p2):
	# get the list of mutually rated items
	si = {}
	for item in prefs[p1]:
		if item in prefs[p2]: 
			si[item] = 1

	# if they are no rating in common, return 0
	if len(si) == 0:
		return 0

	# sum calculations
	n = len(si)

	# sum of all preferences
	sum1 = sum([prefs[p1][it] for it in si])
	sum2 = sum([prefs[p2][it] for it in si])

	# sum of the squares
	sum1Sq = sum([pow(prefs[p1][it],2) for it in si])
	sum2Sq = sum([pow(prefs[p2][it],2) for it in si])

	# sum of the products
	pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

	# calculate r (Pearson score)
	num = pSum - (sum1 * sum2/n)
	den = sqrt((sum1Sq - pow(sum1,2)/n) * (sum2Sq - pow(sum2,2)/n))
	if den == 0:
		return 0

	r = num/den

	return r

# returns the best matches for person from the prefs dictionary
# number of the results and similiraty function are optional params.
def top_matches(prefs,person,n=5,similarity=sim_pearson):
	scores = [(similarity(prefs,person,other),other)
				for other in prefs if other != person]
	scores.sort()
	scores.reverse()
	return scores[0:n]


# gets recommendations for a person by using a weighted average
# of every other user's rankings
def get_recommendations(prefs,person,similarity=sim_pearson):
	totals = {}
	sim_sums = {}

	for other in prefs:
		# don't compare user to itself
		if other == person:
			continue

		# call the similarity function we define
		sim = similarity(prefs,person,other)

		# ignore scores of zero or lower
		if sim <= 0: 
			continue

		# for each of other's items calc totals + sim_sums
		for item in prefs[other]:
			
			# only score items user hasn't seen yet
			if item not in prefs[person] or prefs[person][item] == 0:
				
				# similarity * score
				totals.setdefault(item,0)
				totals[item] += prefs[other][item] * sim
		
				# sum of similarities
				sim_sums.setdefault(item,0)
				sim_sums[item] += sim

	# create the normalized list
	rankings = [(total/sim_sums[item],item) for item,total in totals.items()]

	# return the sorted list
	rankings.sort()
	rankings.reverse()
	return rankings


# function to transform Person, item - > Item, person
def compose_prefs(prefs):
	results = {}
	for person in prefs:
		for item in prefs[person]:
			results.setdefault(item,{})

			# flip item and person
			results[item][person] = prefs[person][item]
	return results

if __name__=="__main__":

	recs = get_recommendations(movies,'Leopoldo Pires')
	print recs