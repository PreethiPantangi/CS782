from algorithms.sasrec.main import SasRec
from algorithms.caser.main import caser

class Recommendation:

    def __init__(self):
        print("In init")

    def sasrec(self):
        SasRec(
            dataset='ml-1m',
            train_dir='default',
            maxlen=200,
            dropout_rate=0.2,
            device='cuda'
        )

    def caser(self):
        caser() 

if __name__ == '__main__':
    recommendation = Recommendation()
    # recommendation.sasrec()
    recommendation.caser()