spark-knn-recommender
===========

Spark-knn-recommender is a fast, scalable recommendation engine built on top of [PySpark](http://spark.apache.org/docs/0.9.0/python-programming-guide.html), the Python API for [Apache Spark](http://spark.apache.org/). It can be deployed locally or on [Amazon EMR](https://aws.amazon.com/elasticmapreduce/).

Requirements
------------

* Python >= 2.7.3
* Spark >= 0.7.0
* Numpy

Installation
------------

First, clone this repo onto your local machine:
```bash
$ git clone https://github.com/evancasey/spark-knn-recommender.git
```

Set your spark cluster configuration in config.py:
```python

# change if you are running on Amazon EMR
CLUSTER_CONFIG = "local"

# fill this in with pySpark path
PYSPARK_HOME = "../spark/pyspark"
SPARKLER_HOME = "../spark/python/sparkler"
```

In the root directory, run:
```bash
$ python setup.py
```

Using spark-knn-recommender
--------------
Run:
```bash
$ python train_and_test.py
```


Running on Amazon EMR
---------------------

* create an Amazan Web Services Account
* sign up for Elastic MapReduce
* install the [Amazon EMR CLI](http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-cli-install.html)

Run the Spark/Shark bootstrap script:
```bash
$ ./elastic-mapreduce --create --alive --name "Spark/Shark Cluster"  --bootstrap-action s3://elasticmapreduce/samples/spark/0.8.1/install-spark-shark.sh --bootstrap-name "Spark/Shark"  --instance-type m1.xlarge --instance-count 3 --jobflow-role spark
Created job flow j-2Y0VECUPLFW94
```

SSH into the master node of your cluster (replace the job ID below with your job ID):

```bash
./elastic-mapreduce -j j-2Y0VECUPLFW94 --ssh
```

## MIT License

Copyright (c) 2011-2016

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
