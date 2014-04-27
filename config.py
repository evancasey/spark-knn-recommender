# local settings
# CLUSTER_CONFIG = "local"
# PYSPARK_HOME = "../build/spark-0.7.0/pyspark"
# PYSPARK_MODULE_HOME = "../build/spark-0.7.0/python/pyspark"
# SPARKLER_HOME = "../build/spark-0.7.0/python/sparkler"

# Amazon EMR settings
CLUSTER_CONFIG = "spark://<master-ip>:7077"
PYSPARK_HOME = "../spark-0.8.1-emr/pyspark"
PYSPARK_MODULE_HOME = "../spark-0.8.1-emr/python/pyspark"
SPARKLER_HOME = "../spark-0.8.1-emr/python/sparkler"