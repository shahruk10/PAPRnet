import numpy as np
import cmath
import cv2

def bin2dec(bits, width, msb='left'):
    chunks = np.reshape(bits, (-1, width))
    decVals = np.zeros(chunks.shape[0])
    for cdx, chunk in enumerate(chunks):
        for i, j in enumerate(chunk):
            if msb == 'right':
                decVals[cdx] += j << i
            else:
                decVals[cdx] += j << (len(chunk)-i-1)
    return decVals


def dec2bin(vals, width, msb='left'):
    if msb == 'left':
        bits = [np.binary_repr(val, width) for val in vals]
    else:
        bits = [np.binary_repr(val, width)[::-1] for val in vals]
    bits = ''.join(bits)
    bits = np.array([int(b) for b in bits])
    return bits

def polar2rect(r, phi):
    phi = cmath.pi * phi / 180.0
    return cmath.rect(r,phi)

def img2bits(img):
    shape = img.shape
    bits = dec2bin(img.flatten(), 8)
    return bits, shape

def bits2img(bitChunks, shape):
    vals = bin2dec(bitChunks, 8)
    img = vals.reshape(shape)
    img = np.uint8(img)
    return img