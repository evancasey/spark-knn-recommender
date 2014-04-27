import pdb
import sys,os

import config
from utils import run_kmeans, run_usercf, run_itemcf
from algorithms import *

DATA_KMEANS = "data/kmeans_data.txt"
DATA_CF_LOCAL = "data/ratings10m.txt"
DATA_CF_S3 = "s3n://sparkler-data/ratings10m.txt"

if __name__ == "__main__":

    # Copy contents of algorithms into pyspark home
    # TODO: use spark_home from install.sh (make install.sh set it in config?)
    os.system("sudo cp -avr algorithms/* " + config.SPARKLER_HOME)

    # run_kmeans(DATA_KMEANS, 2, 5)

    run_usercf(DATA_CF_S3)

    # run_itemcf(DATA_CF_S3)


