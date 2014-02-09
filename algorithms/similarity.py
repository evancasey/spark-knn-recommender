#src: http://aimotion.blogspot.com/2012/08/introduction-to-recommendations-with.html

from math import sqrt


def correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared):
    '''
    The correlation between two vectors A, B is
      [n * dotProduct(A, B) - sum(A) * sum(B)] /
        sqrt{ [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }

    '''
    numerator = size * dot_product - rating_sum * rating2sum
    denominator = sqrt(size * rating_norm_squared - rating_sum * rating_sum) * \
                    sqrt(size * rating2_norm_squared - rating2sum * rating2sum)

    return (numerator / (float(denominator))) if denominator else 0.0


def jaccard(users_in_common, total_users1, total_users2):
    '''
    The Jaccard Similarity between 2 two vectors
        |Intersection(A, B)| / |Union(A, B)|
    '''
    union = total_users1 + total_users2 - users_in_common

    return (users_in_common / (float(union))) if union else 0.0


def normalized_correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared):
    '''
    The correlation between two vectors A, B is
      cov(A, B) / (stdDev(A) * stdDev(B))
      The normalization is to give the scale between [0,1].

    '''
    similarity = correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared)

    return (similarity + 1.0) / 2.0


def cosine(dot_product, rating_norm_squared, rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0


def regularized_correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared,
            virtual_cont, prior_correlation):
    '''
    The Regularized Correlation between two vectors A, B

    RegularizedCorrelation = w * ActualCorrelation + (1 - w) * PriorCorrelation
        where w = # actualPairs / (# actualPairs + # virtualPairs).
    '''
    unregularizedCorrelation = correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared)

    w = size / float(size + virtual_cont)

    return w * unregularizedCorrelation + (1.0 - w) * prior_correlation


def combinations(iterable, r):
    """
    Implementation of itertools combinations method. Re-implemented here because
    of import issues in Amazon Elastic MapReduce. Was just easier to do this than
    bootstrap.
    More info here: http://docs.python.org/library/itertools.html#itertools.combinations

    Input/Output:

    combinations('ABCD', 2) --> AB AC AD BC BD CD
    combinations(range(4), 3) --> 012 013 023 123
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)
