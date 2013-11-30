# Similarity metrics

def cosine(dot_product, rating1_norm_squared, rating2_norm_squared):
    ''' The cosine similarity between two vectors A,B
        dotProduct(A,B) / (norm(A) * norm(B))
    '''

    numerator = dot_product
    denominator = rating1_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0