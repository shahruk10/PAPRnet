import sys
import numpy as np

sys.path.append("..")
from papr import calcPAPR
from plot import plotBER, plotCCDF, plotImages
from utils import dec2bin
from modulation import qamdemod

from PAPRnet import PAPRnetEncoder, PAPRnetDecoder

N = 512

# loading data
x = np.load('data/modChunks.npz')['data']
xval = x[:1000]
xvalrect = np.concatenate( [np.expand_dims(xval.real, 1), np.expand_dims(xval.imag, 1)], axis=1)    # splitting real and imaginary comp

# Encoding
encoder = PAPRnetEncoder(N)
encoder.load_weights('encoder.hdf5')

xenc = encoder.predict(xvalrect, batch_size=512)   # getting encoder output
xhidd = xenc[:, 0, :] + 1j * xenc[:, 1, :]         # converting to complex vals (a+bj)

# Decoding
decoder = PAPRnetDecoder(N)
decoder.load_weights('decoder.hdf5')

xdec = decoder.predict(xenc, batch_size=512)
xest = xdec[:, 0, :] + 1j * xdec[:, 1, :]

# Calculating BER and PAPR
papr = []

# getting bits from original signal
xsig = np.fft.ifft(xval, n=512, axis=-1)
bits = dec2bin(qamdemod(xval, 4).flatten(), 2)
papr.append(calcPAPR(xsig))

# calculating PAPR of encoder output 
xhiddSig = np.fft.ifft(xhidd, n=512, axis=-1)
papr.append(calcPAPR(xhiddSig))

# getting bits from decoder output
xestSig = np.fft.ifft(xest, n=512, axis=-1)
rxNN = np.fft.fft(xestSig, axis=-1)
rxbits = dec2bin(qamdemod(rxNN,4).flatten(),2)

# calculating BER
BER = np.sum(np.logical_xor(rxbits, bits)) / (1.0 * len(bits))
print("BER = {}".format(BER))

plotCCDF(papr, steps=0.25, savePath='./NN_PAPR.png',
        title="PAPR Reduction using Encoder-Decoder Model", 
        legend=['Original', 'NN Encoded'])
