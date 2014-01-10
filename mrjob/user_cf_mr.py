''' User-based Collaborative Filtering '''

from mrjob.job import MRJob
from metrics import correlation, jaccard, cosine, regularized_correlation
from math import sqrt
import pdb

try:
    from itertools import combinations
except ImportError:
    from metrics import combinations


PRIOR_COUNT = 10
PRIOR_CORRELATION = 0


class CommaValueProtocol(object):

    def write(self, key, values):
        out_values = [str(v).replace("]","") \
                            .replace("[","") \
                            .replace("set(","") \
                            .replace(")","") \
                            .replace("'","") \
                            .replace(" ","") for v in values]
        return ','.join(str(v) for v in out_values)


class UserCF(MRJob):

    OUTPUT_PROTOCOL = CommaValueProtocol

    def steps(self):
        return [
            self.mr(mapper=self.group_by_item_rating,
                    reducer=self.get_item_hist),
            self.mr(mapper=self.map_to_item_id,
                    reducer=self.count_ratings_items_freq),
            self.mr(mapper=self.pairwise_users,
                    reducer=self.calculate_similarity),
            self.mr(mapper=self.calculate_ranking,
                    reducer=self.top_recommendations),
            self.mr(mapper=self.identity,
                    reducer=self.group_by_user)]

    def group_by_item_rating(self, key, line):

        user_id, item_id, rating = line.split(',')        
        yield  user_id, (item_id, float(rating))

    def get_item_hist(self, user_id, values):

        item_hist = []

        for item_id, rating in values:
            item_hist.append((item_id,rating))

        yield user_id, item_hist

    def map_to_item_id(self, user_id, item_hist):

        for (item_id, rating) in item_hist: 
            yield item_id, (user_id,item_hist,rating)

    def count_ratings_items_freq(self, item_id, values):

        user_count = 0
        user_sum = 0
        final = []
        for user_id, item_hist, rating in values:
            user_count += 1
            user_sum += rating
            final.append(((user_id, item_hist), rating))

        yield item_id, (user_count, user_sum, final)

    def pairwise_users(self, item_id, values):

        user_count, user_sum, ratings = values

        for user1, user2 in combinations(ratings, 2):
            yield (user1[0], user2[0]), \
                    (user1[1], user2[1], item_id)

    def calculate_similarity(self, pair_key, lines):
    
        sum_xx, sum_xy, sum_yy, n = (0.0, 0.0, 0.0, 0)
        user_pair, co_ratings_id = pair_key, lines
        user_xname, user_yname = user_pair
        for user_x, user_y, item_id in lines:
            sum_xx += user_x * user_x
            sum_yy += user_y * user_y
            sum_xy += user_x * user_y
            n += 1

        cos_sim = cosine(sum_xy, sqrt(sum_xx), sqrt(sum_yy))

        yield (user_xname, user_yname), (cos_sim, n)

    def calculate_ranking(self, user_keys, values):

        cos_sim, n = values
        user_x, user_y = user_keys
        if int(n) > 0:
            yield (user_x, cos_sim), \
                     (user_y, n)

    def top_recommendations(self, key_sim, similar_ns):

        user_x, cos_sim = key_sim

        for user_y, n in similar_ns:

            totals = {}
            sim_sums = {}
            
            if user_y[0] != user_x[0]:
                for rating in user_y[1]:
                    items_x = [x[0] for x in user_x[1]]
                    if rating[0] not in items_x:
                        # if not, compute totals and sim_sums for each item
                        totals.setdefault(rating[0],0)
                        totals[rating[0]] += cos_sim * rating[1]

                        sim_sums.setdefault(rating[0],0)
                        sim_sums[rating[0]] += cos_sim

                # create the normalized list
                rankings = [(total/sim_sums[item],item) for item,total in totals.items()]

                # return the sorted list
                rankings.sort()
                rankings.reverse()        

                top_recs = [x[1] for x in rankings]

            yield user_x[0],top_recs

    def identity(self, user_id, rankings):
        yield user_id, rankings

    def group_by_user(self, user_id, rankings):

        all_recs = []
        for r in rankings:
            all_recs.append(r)

        recs = set()
        while len(recs) <= 500 and len(all_recs) > 0:
            recs.update(all_recs.pop(0))
        yield None, recs


if __name__ == '__main__':
    UserCF.run()
