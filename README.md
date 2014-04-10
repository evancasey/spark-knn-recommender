Sparkler
===========

Sparkler is a Python library for building scalable recommender systems. Sparkler is built on top of [PySpark](http://spark.apache.org/docs/0.9.0/python-programming-guide.html), the Python API for [Apache Spark](http://spark.apache.org/).

Requirements
------------

* Python >= 2.7.3
* Spark >= 0.7.0
* Numpy

Installation
------------

First, clone sparkler onto your local machine:
```bash
$ git clone https://github.com/evancasey/sparkler.git
```

Configure sparkler in config.py:
```python

# change if you are running on Amazon EMR
CLUSTER_CONFIG = "local"

# fill this in with pySpark path
PYSPARK_HOME = "../spark/pyspark"
SPARKLER_HOME = "../spark/python/sparkler"
```

In your sparkler directory, run:
```bash
$ python setup.py
```

Using sparkler
--------------



Running on Amazon EMR
---------------------

* create an Amazan Web Services Account
* sign up for Elastic MapReduce
* Acquire secret keys
* Run Spark/Shark bootstrap script


