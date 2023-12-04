from algorithms.sasrec.main import SasRec
from algorithms.caser.main import caser
from algorithms.Bert4Rec.bert4rec import train
from algorithms.Bert4Rec.runoptions import args as runoptions_args
from algorithms.Bert4Rec.templates import set_template
from datapreprocessing.datapreprocessing import datapreprocessing

# Martians: Library for Sequential Recommendation algorithms SASRec, Caser, and Bert4Rec
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
    
    def bert4rec(self, datasetName):
        datasetTocall = '1' if datasetName == 'MovieLens' else '2'
        import argparse

        # Define an argparse parser to parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--template', type=str, default='train_bert')
        parser.add_argument('--dataset_code', type=str, default=datasetTocall)
        parser.add_argument('--train_negative_sampling_seed', type=int, default=0)
        parser.add_argument('--enable_lr_schedule', type=bool, default=True)

        args = parser.parse_args(namespace=runoptions_args)
        set_template(args)
        train() 

if __name__ == '__main__':
    dataset = 'beauty'
    # Martians: Pre-processing data based on the dataset name.
    datapreprocessing(dataset=dataset)
    recommendation = Recommendation()
    recommendation.sasrec(dataset)
    # recommendation.caser(dataset)
    # recommendation.bert4rec(dataset)