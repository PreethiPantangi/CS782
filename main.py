from algorithms.sasrec.main import SasRec
from algorithms.caser.main import caser
from datapreprocessing.datapreprocessing import datapreprocessing

class Recommendation:

    def __init__(self):
        print("In init")

    def sasrec(self, dataset):
        SasRec(
            dataset=dataset,
            train_dir='default',
            maxlen=200,
            dropout_rate=0.2,
            device='cuda'
        )

    def caser(self, dataset):
        caser(dataset) 

if __name__ == '__main__':
    dataset = 'beauty'
    datapreprocessing(dataset=dataset)
    recommendation = Recommendation()
    recommendation.sasrec(dataset)
    # recommendation.caser(dataset)