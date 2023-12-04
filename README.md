# Advanced Machine Learning - CS782 - Martians

Sequential RecSys is an open-Source PyTorch based library, inspired by the structure of the widely used Scikit-learn library. The library encompasses three essential sequential recommendation algorithms: SASRec (Self-Attentive Sequential Recommendation), Caser (Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding), and Bert4Rec (Sequential Recommendation with Bidirectional Encoder Representations from Transformer). In addition to facilitating loading and pre-processing the data, the library will incorporate a ranking-based recommendation evaluation module, featuring Hit@k and NDCG@k metrics.

## Datasets

Movie Lens 1M - https://grouplens.org/datasets/movielens/1m/ <br/>

Amazon Beauty - https://jmcauley.ucsd.edu/data/amazon/index.html

## Requirements

Python 2 or 3 <br/>
PyTorch v0.4+ <br/>
Numpy <br/>
SciPy <br/>
wget <br/>
tqdm <br/>
torch <br/>
tb-nightly <br/>
pandas <br/>
scipy <br/>
future <br/>

## Command to execute the code 

python main.py

## Using Recommendation class in your code
In case you would like to use the module in your code you can call the desired algorithm (sasrec, caser, or bert4rec) on Recommendation class from main.py and specify the dataset name (ml-1m or beauty) <br/>
Ex: Recommendation.sasrec('beauty') <br/>
Ex: Recommendation.caser('ml-1m') <br/>
Ex: Recommendation.bert4rec('ml-1m') <br/>

### Caser - Plot for HIT and NDCG vs Epoch
![caser_metrics](https://github.com/PreethiPantangi/CS782/assets/22561209/5f41d5ae-44a8-4601-a2ba-b1186bdde219)

### NDCG - Plot for NDCG vs Epoch
![bert4rec_metrics](https://github.com/PreethiPantangi/CS782/assets/22561209/692bb8a8-0c9c-4ba6-9c08-03d867a66018)
