/** src: http://mlnick.github.io/blog/2013/04/01/movie-recommendations-and-more-with-spark/
  */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object ItemCF {

  def main(args: Array[String]) {

    // initialize the spark context
    val sc = new SparkContext("local", "UserCF", "/home/evan/dev/grasshopper",
      List("target/scala-2.9.3/grasshopper_2.9.3-1.0.jar"))

    // specify the data we want to use
    val TRAIN_FILENAME = "ua.base"
    val MOVIES_FILENAME = "u.item"

    // extract userid, movieid, rating from ratings data
    val ratings = sc.textFile("data/ml-100k/" + TRAIN_FILENAME)
      .map(line => {
      val fields = line.split("\t")
      (fields(0).toInt, fields(1).toInt, fields(2).toInt)
    })

    // get num raters per movie, keyed on movie id
    val numRatersPerMovie = ratings
      .groupBy(tup => tup._2)
      .map(grouped => (grouped._1, grouped._2.size))

    // join ratings with num raters on movie id
    val ratingsWithSize = ratings
      .groupBy(tup => tup._2)
      .join(numRatersPerMovie)
      .flatMap(joined => {
        joined._2._1.map(f => (f._1, f._2, f._3, joined._2._2))
    })

    // ratingsWithSize now contains the fields: (user, movie, rating, numRaters)

    // dummy copy of ratings for self join
    val ratings2 = ratingsWithSize.keyBy(tup => tup._1)

    // join on userid and filter movie pairs such that we don't double-count and exclude self-pairs
    val ratingPairs =
      ratingsWithSize
      .keyBy(tup => tup._1)
      .join(ratings2)
      .filter(f => f._2._1._2 < f._2._2._2)

    // compute raw inputs to similarity metrics for each movie pair
    val vectorCalcs =
      ratingPairs
        .map(data => {
        val key = (data._2._1._2, data._2._2._2)
        val stats =
          (data._2._1._3 * data._2._2._3, // rating 1 * rating 2
            data._2._1._3,                // rating movie 1
            data._2._2._3,                // rating movie 2
            math.pow(data._2._1._3, 2),   // square of rating movie 1
            math.pow(data._2._2._3, 2),   // square of rating movie 2
            data._2._1._4,                // number of raters movie 1
            data._2._2._4)                // number of raters movie 2
        (key, stats)
      })
        .groupByKey()
        .map(data => {
        val key = data._1
        val vals = data._2
        val size = vals.size
        val dotProduct = vals.map(f => f._1).sum
        val ratingSum = vals.map(f => f._2).sum
        val rating2Sum = vals.map(f => f._3).sum
        val ratingSq = vals.map(f => f._4).sum
        val rating2Sq = vals.map(f => f._5).sum
        val numRaters = vals.map(f => f._6).max
        val numRaters2 = vals.map(f => f._7).max
        (key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2))
      })

    val PRIOR_COUNT = 10
    val PRIOR_CORRELATION = 0

    // compute similarity metrics for each movie pair
    val similarities =
      vectorCalcs
        .map(fields => {

        val key = fields._1
        val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields._2

        val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
        val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum,
          ratingNormSq, rating2NormSq, PRIOR_COUNT, PRIOR_CORRELATION)
        val cosSim = cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))
        val jaccard = jaccardSimilarity(size, numRaters, numRaters2)

        (key, (corr, regCorr, cosSim, jaccard))
      })
  }


  def regularizedCorrelation(size : Double, dotProduct : Double, ratingSum : Double,
                             rating2Sum : Double, ratingNormSq : Double, rating2NormSq : Double,
                             virtualCount : Double, priorCorrelation : Double) = {

    val unregularizedCorrelation = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
    val w = size / (size + virtualCount)

    w * unregularizedCorrelation + (1 - w) * priorCorrelation
  }


  def correlation(size : Double, dotProduct : Double, ratingSum : Double,
                  rating2Sum : Double, ratingNormSq : Double, rating2NormSq : Double) = {

    val numerator = size * dotProduct - ratingSum * rating2Sum
    val denominator = math.sqrt(size * ratingNormSq - ratingSum * ratingSum) * math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum)

    numerator / denominator
  }


  def cosineSimilarity(dotProduct : Double, ratingNorm : Double, rating2Norm : Double) = {
    dotProduct / (ratingNorm * rating2Norm)
  }


  def jaccardSimilarity(usersInCommon : Double, totalUsers1 : Double, totalUsers2 : Double) = {
    val union = totalUsers1 + totalUsers2 - usersInCommon
    usersInCommon / union
  }

}
