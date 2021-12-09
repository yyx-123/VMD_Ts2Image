import os
from train import train

datasetPath = "../dataset/images/"
for dataset in os.listdir(datasetPath):
    datasetName = dataset.split('.')[0]
    train(datasetName)
