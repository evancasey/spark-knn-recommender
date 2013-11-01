/** Implementation of user-based collaborative filtering with Cosine Similarity
	* and Pearson correlation coefficient similarity measures using the 
	* Movielens 100k dataset
	*/

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object UserCF {

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

  def main(args: Array[String]) {

  	val sim = sim_cosine(_:List[(Int,Int)], _:List[(Int,Int)]) // set similarity measure
		val user = 3 // the user we want to recommend items for

		// get recommendations for this user
		val recs = getRecs(user,sim)

	}

	// returns a list of n recommendations for the user
	def getRecs(user: Int, sim: (List[(Int,Int)], List[(Int,Int)]) => Int): Int = {

		// exclude the user we are looking at
		val filtRatings = ratings.filter(_._1 != user)


		// transform the filtered ratings into a map of users and movie,rating pairs
		// where each pair is a movie that the active user and another
		// user in that dataset has rated (duplicates will exist)
		val ratingPairs = "FILL IN"


		//call sim_cosine on each movie,rating pair
		val itemSims = "FILL IN"
		2
	}

	// returns a the cosine similarity for two users
	// accepts two movie user
	def sim_cosine(p1: List[(Int,Int)], p2: List[(Int,Int)]): Int = {
		// FILL IN
		3
	}

}