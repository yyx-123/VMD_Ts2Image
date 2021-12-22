import matplotlib.pyplot as plt
import numpy as np
import os

def calRecall(confusion):
    recall0 = confusion[0][0] / confusion[0].sum()
    recall1 = confusion[1][1] / confusion[1].sum()
    recall2 = confusion[2][2] / confusion[2].sum()
    return recall0, recall1, recall2

def calPrecision(confusion):
    prec0 = confusion[0][0] / (confusion[0][0] + confusion[1][0] + confusion[2][0])
    prec1 = confusion[1][1] / (confusion[0][1] + confusion[1][1] + confusion[2][1])
    prec2 = confusion[2][2] / (confusion[0][2] + confusion[1][2] + confusion[2][2])
    return prec0, prec1, prec2
