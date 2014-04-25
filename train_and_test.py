import pdb
import sys,os

import config
from utils import run_kmeans, run_usercf, run_itemcf, run_user_itemcf, run_user_item_cf_broadcast
from algorithms import *

DATA_KMEANS = "data/kmeans_data.txt"
DATA_USERCF = "data/ratings1.txt"
DATA_ITEMCF = "s3n://sparkler-data/ratings10k.txt"

if __name__ == "__main__":

    # Copy contents of algorithms into pyspark home
    # TODO: use spark_home from install.sh (make install.sh set it in config?)
    os.system("sudo cp -avr algorithms/* " + config.SPARKLER_HOME)

    # run_kmeans(DATA_KMEANS, 2, 5)

    # run_usercf(DATA_USERCF)

    # run_itemcf(DATA_ITEMCF)

    run_user_itemcf(DATA_USERCF)
   
    # run_user_item_cf_broadcast(DATA_ITEMCF)

