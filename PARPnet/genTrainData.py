import numpy as np
import cv2
import glob
import sys

sys.path.append("..")
from ofdm import OFDMTransmitter, OFDMReceiver
from channel import SISOFlatChannel, awgn
from papr import calcPAPR
from plot import plotBER, plotCCDF, plotImages
from utils import polar2rect, img2bits, bits2img

def main():

    params = {}
    params['N'] = 512
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1
    params['dumpModChunks'] = True

    imagePaths = sorted(glob.glob('../sample_data/*.jpg'))

    modChunks = []
    for imgp in imagePaths:
        img = cv2.imread(imgp)
        bitStream, imgShape = img2bits(img)
        
        tx = OFDMTransmitter(**params)
        modChunks.append(tx.transmit(bitStream))

    # print(modChunks[0].shape)
    modChunks = np.concatenate(modChunks, axis=0)
    print(modChunks.shape)
    np.savez('modChunks.npz', data=modChunks)

if __name__ == "__main__":
    main()