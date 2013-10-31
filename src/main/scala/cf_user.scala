/** Implementation of user-based collaborative filtering with Cosine Similarity
	* and Pearson correlation coefficient similarity measures using the Movielens 100k dataset
	*/

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object UserCF {
  def main(args: Array[String]) {

  	// initialize the spark context
    val sc = new SparkContext("local", "UserCF", "/home/evan/dev/grasshopper",
      List("target/scala-2.9.3/grasshopper_2.9.3-1.0.jar"))

    val TRAIN_FILENAME = "ua.base"
    val MOVIES_FILENAME = "u.item"

    // extract userid, movieid, rating from ratings data
		val ratings = sc.textFile("data/ml-100k/" + TRAIN_FILENAME)
			.map(line => {
				val fields = line.split("\t")
				(fields(0).toInt, fields(1).toInt, fields(2).toInt)
			})

		// dummy copy of ratings for self join
		val ratings2 = ratings.keyBy(tup => tup._1)

		// join on userid and filter movie pairs such that we don't double-count and exclude self-pairs
		val ratingPairs = ratings
		  .keyBy(tup => tup._1)
		  .join(ratings2)
		  .filter(f => f._2._1._2 < f._2._2._2)



		// // select which similarity metric
		// val sim_type = "sim_cosine"

		// // select which person we want prefs for
		// val users = List("Luciana Nunes")

		// /** 
		//  * gets recommendations for a person by using a weighted average
		//  * of every other user's rankings
		//  */
		// val prefs = ratings.join('person)

		/**
		 * to get all pairs of co-rated items, we'll join 'ratings' against itself.
		 * So first make a dummy copy of the ratings that we can join against.
		 */
		// val si = ratings.k

		// println(si)
		// ._2._1.join(_._2._1)


		// val totals

		// val sim_sums

		// println(ratings.length)

  }
}