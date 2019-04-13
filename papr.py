import numpy as np
import matplotlib.pyplot as plt

def calcPAPR(sig):
    
    sigSq = np.power(np.abs(sig),2)
    paprDb = 10.0 * np.log10(np.divide(np.max(sigSq, axis=-1),np.mean(sigSq, axis=-1))) 

    return paprDb
