IMPORTANT:

If you don't want to use the IMDB data already included in this directory,
please move all the scripts into the aclImdb directory(the directory after uncompressed)

To preprocess IMDB data, run:

python preprocessing.py

To run experiments dealing with 20 news dataset, run:

python 20_news.py

This script takes very long time to run, but you can directly see the result in 20news_result.txt

(After running preprocessing.py)To run experiments dealing with IMDB dataset, run:

python IMDB.py

This script takes very long time to run, but you can directly see the result in IMDB result.txt

when test.csv and train.csv exist, you can directly run IMDB.py without preprocessing.py