import numpy as np
import cv2

from ofdm import OFDMTransmitter, OFDMReceiver
from channel import SISOFlatChannel, awgn
from papr import calcPAPR
from plot import plotBER, plotCCDF, plotImages
from utils import polar2rect, img2bits, bits2img
from keras.models import load_model

def N_vs_PAPR():

    print("------ Running # of Sub Carrier vs PAPR Simulation -------")
    params = {}
    params['N'] = 64
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1

    # nBits = 2**20
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)
    
    paprDb = []

    Nrange = [64, 128, 256, 512, 1024, 2048]
    for N in Nrange:
        params['N'] = N

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.padBits = tx.padBits
        rbitStream = rx.receive(sig.flatten())
        BER = np.sum(np.logical_xor(rbitStream, bitStream)) /(1.0 * len(bitStream))

        print(" N = {:<3}  Max PAPR (dB) = {:<5.3}  BER = {:<5.3}".format(
            N, np.max(paprDb[-1]), BER))

    legend = ['N = {}'.format(val) for val in Nrange]
    plotCCDF(paprDb, savePath='N_vs_PAPR.png', show=False, steps=0.25,
             title='Complementary Cumulative Distribution Function (CCDF) for PAPR\nRelationship with Number of Sub Carriers', legend=legend)

    print("".join(["-"]*60), "\n\n")

def L_vs_PAPR():

    print("------ Running Upsampling Factor vs PAPR Simulation -------")
    params = {}
    params['N'] = 512
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1

    # nBits = 2**16
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []

    upsampleFactorRange = [1, 2, 4, 8, 16]
    for L in upsampleFactorRange:
        params['upsampleFactor'] = L

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.padBits = tx.padBits
        rbitStream = rx.receive(sig.flatten())
        BER = np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream))

        print(" Upsampling factor = {:<3}  Max PAPR (dB) = {:<5.3}  BER = {:<5.3}".format(L, np.max(paprDb[-1]), BER))

    legend = [ 'L = {}'.format(val) for val in upsampleFactorRange]
    plotCCDF(paprDb, savePath='L_vs_PAPR.png', show=False, steps=0.25,
             title='Complementary Cumulative Distribution Function (CCDF) for PAPR\nRelationship with Upsampling Factor', legend=legend)

    print("".join(["-"]*60), "\n\n")

def N_vs_BER():

    print("------ Running # of Sub Carrier vs BER Simulation -------")
    params = {}
    params['N'] = 64
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1

    # nBits = 2**16
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []
    BERSets = []

    Nrange = [64, 128, 256, 512, 1024, 2048]
    snrRange = np.arange(-10,40,1)

    for N in Nrange:
        params['N'] = N

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.padBits = tx.padBits

        BER = []
        for snr in snrRange:
            # applying channel with fading
            avgOFDMSymbolPower = np.mean(np.mean(np.power(np.abs(sig), 2),axis=-1))
            channel = SISOFlatChannel(fading_param=( polar2rect(0.9,15.0), 0.19))
            channel.set_SNR_dB(snr, Es=avgOFDMSymbolPower)
            noisySig = channel.propagate(sig.flatten())
            
            # decoding
            rbitStream = rx.receive(noisySig)
            BER.append(np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream)))
            print(" N = {:<6} SNR (dB) = {:<4}  BER = {:<5.3}".format(N, snr, BER[-1]))

        BERSets.append(BER)


    legend = ['N = {}'.format(val) for val in Nrange]
    plotBER(BERSets, snrRange, xlabel='SNR (dB)', savePath='N_vs_BER.png', show=False, semilog=True,
             title='Bit Error Rate in Rician Channel\nRelationship with Number of Sub Carriers', legend=legend)

    print("".join(["-"]*60), "\n\n")

def NCP_vs_BER():

    print("------ Running Cyclic Prefix vs BER Simulation -------")
    params = {}
    params['N'] = 512
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 16
    params['upsampleFactor'] = 1

    # nBits = 2**16
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []
    BERSets = []

    NCPrange = [0.10, 0.25, 0.35, 0.5, 0.75]
    snrRange = np.arange(-10,40,1)

    for NCP in NCPrange:
        params['cyclicPrefix'] = int(NCP * params['N'])

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.padBits = tx.padBits

        BER = []
        for snr in snrRange:
            # applying channel with fading
            avgOFDMSymbolPower = np.mean(np.mean(np.power(np.abs(sig), 2),axis=-1))
            channel = SISOFlatChannel(fading_param=( polar2rect(0.9,30.0), 0.19))
            channel.set_SNR_dB(snr, Es=avgOFDMSymbolPower)
            noisySig = channel.propagate(sig.flatten())
            
            # decoding
            rbitStream = rx.receive(noisySig)
            BER.append(np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream)))
            print(" NCP = {:<6} SNR (dB) = {:<4}  BER = {:<5.3}".format(NCP, snr, BER[-1]))

        BERSets.append(BER)


    legend = ['CP = {}'.format(val) for val in NCPrange]
    plotBER(BERSets, snrRange, xlabel='SNR (dB)', savePath='NCP_vs_BER.png', show=False, semilog=True,
             title='Bit Error Rate in Rician Channel\nRelationship with Cyclic Prefix', legend=legend)

    print("".join(["-"]*60), "\n\n")

def L_vs_BER():

    print("------ Running Upsampling Factor vs BER Simulation -------")
    params = {}
    params['N'] = 512
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1

    # nBits = 2**16
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []
    BERSets = []

    snrRange = np.arange(-10,40,1)

    upsampleFactorRange = [1, 2, 4, 8, 16]
    for L in upsampleFactorRange:
        params['upsampleFactor'] = L

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.padBits = tx.padBits

        BER = []
        for snr in snrRange:
            # applying channel with fading
            avgOFDMSymbolPower = np.mean(np.mean(np.power(np.abs(sig), 2),axis=-1))
            channel = SISOFlatChannel(fading_param=( polar2rect(0.9,10.0), 0.19))
            channel.set_SNR_dB(snr, Es=avgOFDMSymbolPower)
            noisySig = channel.propagate(sig.flatten())
            
            # decoding
            rbitStream = rx.receive(noisySig)
            BER.append(np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream)))
            print(" L = {:<6} SNR (dB) = {:<4}  BER = {:<5.3}".format(L, snr, BER[-1]))

        BERSets.append(BER)


    legend = ['L = {}'.format(val) for val in upsampleFactorRange]
    plotBER(BERSets, snrRange, xlabel='SNR (dB)', savePath='L_vs_BER.png', show=False, semilog=True,
             title='Bit Error Rate in Rician Channel\nRelationship with Upsampling Factor', legend=legend)

    print("".join(["-"]*60), "\n\n")

def SLM_vs_PAPR():
    print("------ Running SLM Candidates vs PAPR Simulation -------")
    params = {}
    params['N'] = 512
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1
    params['useSLM'] = True

    # nBits = 2**20
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []

    phaseCandidates = [0, 8, 16, 32, 64]
    for C in phaseCandidates:
        params['SLMCandidates'] = C

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.padBits = tx.padBits
        rx.SLMPhaseVectorIdx = tx.SLMPhaseVectorIdx
        rbitStream = rx.receive(sig.flatten())
        BER = np.sum(np.logical_xor(rbitStream, bitStream)) /(1.0 * len(bitStream))

        print(" C = {:<3}  Max PAPR (dB) = {:<5.3}  BER = {:<5.3}".format(
            C, np.max(paprDb[-1]), BER))

    legend = ['Candidates = {}'.format(val) for val in phaseCandidates]
    legend[0] = 'No SLM'
    plotCCDF(paprDb, savePath='SLM-C_vs_PAPR_L-{}.png'.format(params['upsampleFactor']), show=False,
             legend=legend,
             title='Complementary Cumulative Distribution Function (CCDF) for PAPR\nApplying Selective Mapping for N = {} and L = {}'.format(params['N'], params['upsampleFactor']))

    print("".join(["-"]*60), "\n\n")

def Clipping_vs_PAPR():
    print("------ Running Clipping vs PAPR Simulation -------")
    params = {}
    params['N'] = 512
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1
    params['useClipping'] = True

    # nBits = 2**20
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []
    BERSets = []

    snrRange = np.arange(-10, 20, 1)
    
    clippingFactor = [1.0, 0.9, 0.75, 0.5]
    for C in clippingFactor:
        params['clippingPercent'] = C

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.padBits = tx.padBits
        BER = []
        for snr in snrRange:
            # applying channel with fading
            avgOFDMSymbolPower = np.mean(np.mean(np.power(np.abs(sig), 2),axis=-1))
            channel = SISOFlatChannel(fading_param=( polar2rect(0.9,10.0), 0.19))
            channel.set_SNR_dB(snr, Es=avgOFDMSymbolPower)
            noisySig = channel.propagate(sig.flatten())

            # decoding
            rbitStream = rx.receive(noisySig)

            BER.append(np.sum(np.logical_xor(rbitStream, bitStream)) /(1.0 * len(bitStream)))
            print(" C = {:<3.1f}  Max PAPR (dB) = {:<5.3} SNR(db) = {:<4} BER = {:<5.3}".format(float(C*100.0), np.max(paprDb[-1]), snr, BER[-1]))

        BERSets.append(BER)

    legend = ['Clipping = {:3.1f} %% of Max'.format(float(val*100.0)) for val in clippingFactor]
    legend[0] = 'No Clipping'
    plotCCDF(paprDb, savePath='Clipping_vs_PAPR_L-{}.png'.format(params['upsampleFactor']), show=False,
             steps=0.25, legend=legend,
             title='Complementary Cumulative Distribution Function (CCDF) for PAPR\nApplying Clipping for N = {} and L = {}'.format(params['N'], params['upsampleFactor']))
    
    plotBER(BERSets, snrRange, xlabel='SNR (dB)', savePath='Clipping_vs_BER_L-{}.png'.format(params['upsampleFactor']), show=False, semilog=True,
            title='Bit Error Rate in Rician Channel\nRelationship with Clipping OFDM Signal', legend=legend)

    print("".join(["-"]*60), "\n\n")

def ClippingSLM_vs_PAPR():
    print("------ Running Clipping vs PAPR Simulation -------")
    params = {}
    params['N'] = 512
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1
    params['useClipping'] = True
    params['clippingPercent'] = 0.75
    params['useSLM'] = True
    params['SLMCandidates'] = 32

    # nBits = 2**20
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []
    BERSets = []
    legend = []

    snrRange = np.arange(-10, 40, 1)

    for useClipping, useSLM in  zip([False, True, False, True], [False, False, True, True]) :
        
        params['useSLM'] = useSLM
        params['useClipping'] = useClipping
        legend.append('Clipping = {} SLM = {}'.format(useClipping, useSLM))

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.padBits = tx.padBits
        rx.SLMPhaseVectorIdx = tx.SLMPhaseVectorIdx
        
        BER = []
        for snr in snrRange:
            # applying channel with fading
            avgOFDMSymbolPower = np.mean(np.mean(np.power(np.abs(sig), 2),axis=-1))
            channel = SISOFlatChannel(fading_param=( polar2rect(0.9,10.0), 0.19))
            channel.set_SNR_dB(snr, Es=avgOFDMSymbolPower)
            noisySig = channel.propagate(sig.flatten())

            # decoding
            rbitStream = rx.receive(noisySig)

            BER.append(np.sum(np.logical_xor(rbitStream, bitStream)) /(1.0 * len(bitStream)))
            print(" SLM = {} Clipping = {} Max PAPR (dB) = {:<5.3} SNR(db) = {:<4} BER = {:<5.3}".format(useSLM, useClipping, np.max(paprDb[-1]), snr, BER[-1]))

        BERSets.append(BER)


    legend[0] = 'No SLM or Clipping'
    plotCCDF(paprDb, savePath='ClippingSLM_vs_PAPR.png', show=False,
             steps=0.25, legend=legend,
             title='Complementary Cumulative Distribution Function (CCDF) for PAPR\nUsing Clipping and SLM')
    
    plotBER(BERSets, snrRange, xlabel='SNR (dB)', savePath='ClippingSLM_vs_BER.png', show=False, semilog=True,
            title='Bit Error Rate in Rician Channel\nRelationship with Clipping OFDM Signal', legend=legend)

    print("".join(["-"]*60), "\n\n")

def ConvCoding_vs_PAPR():
    print("------ Running Conv Coding vs PAPR Simulation -------")
    params = {}
    params['N'] = 512
    params['modType'] = 'qam'
    params['M'] = 4
    params['useConvCode'] = True

    # nBits = 2**14
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/ece.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []
    legend = []

    ConvCodeGenerators = [[], [0o5, 0o7], [0o5, 0o7, 0o3], [0o5, 0o7, 0o3, 0o06]]   # generator polynomial connections
    for G in ConvCodeGenerators:
        params['convCodeGMatrix'] = np.array(G)

        tx = OFDMTransmitter(**params)
        sig = tx.transmit(bitStream)
        paprDb.append(calcPAPR(sig))

        rx = OFDMReceiver(**params)
        rx.trellis = tx.trellis
        rx.padBits = tx.padBits
        rbitStream = rx.receive(sig.flatten())
        BER = np.sum(np.logical_xor(rbitStream, bitStream)) /(1.0 * len(bitStream))

        legend.append('Rate = {}'.format(tx.codeRate))

        print(" CodeRate = {}  Max PAPR (dB) = {:<5.3}  BER = {:<5.3}".format(tx.codeRate, np.max(paprDb[-1]), BER))

    legend[0] = 'No Conv Coding'
    plotCCDF(paprDb, savePath='ConvCoding_vs_PAPR.png', show=False, steps=0.25,
             legend=legend,
             title='Complementary Cumulative Distribution Function (CCDF) for PAPR\nApplying Convolutional Coding')

    print("".join(["-"]*60), "\n\n")

def SLM_vs_NN():
    print("------ Running SLM vs NN Simulation -------")

    params = {}
    params['N'] = 512
    params['cyclicPrefix'] = int(0.25 * params['N'])
    params['modType'] = 'qam'
    params['M'] = 4
    params['upsampleFactor'] = 1
  

    # nBits = 2**20
    # bitStream = np.random.randint(0, 2, nBits, dtype=np.uint8)
    img = cv2.imread('sample_data/wild.jpg')
    bitStream, imgShape = img2bits(img)

    paprDb = []
    legend = []

    # Normal OFDM ------------------------------------------
    tx = OFDMTransmitter(**params)
    sig = tx.transmit(bitStream)
    paprDb.append(calcPAPR(sig))

    rx = OFDMReceiver(**params)
    rx.padBits = tx.padBits
    rbitStream = rx.receive(sig.flatten())
    BER = np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream))

    legend.append("Normal OFDM")

    # OFDM with SLM 32 ------------------------------------------
    params['useSLM'] = True
    params['SLMCandidates'] = 32
    tx = OFDMTransmitter(**params)
    sig = tx.transmit(bitStream)
    paprDb.append(calcPAPR(sig))

    rx = OFDMReceiver(**params)
    rx.padBits = tx.padBits
    rx.SLMPhaseVectorIdx = tx.SLMPhaseVectorIdx
    rbitStream = rx.receive(sig.flatten())
    BER = np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream))
    print("SLM 32 : BER = {:<5.2}".format(BER))
    legend.append("OFDM + SLM-32")

    # OFDM with SLM 64 ------------------------------------------
    params['useSLM'] = True
    params['SLMCandidates'] = 64
    tx = OFDMTransmitter(**params)
    sig = tx.transmit(bitStream)
    paprDb.append(calcPAPR(sig))

    rx = OFDMReceiver(**params)
    rx.padBits = tx.padBits
    rx.SLMPhaseVectorIdx = tx.SLMPhaseVectorIdx
    rbitStream = rx.receive(sig.flatten())
    BER = np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream))
    print("SLM 64 : BER = {:<5.2}".format(BER))
    legend.append("OFDM + SLM-64")

    # OFDM with SLM 64 and Clipping ------------------------------------------
    params['useSLM'] = True
    params['SLMCandidates'] = 64
    params['useClipping'] = True
    params['clippingPercent'] = 0.75
    tx = OFDMTransmitter(**params)
    sig = tx.transmit(bitStream)
    paprDb.append(calcPAPR(sig))

    rx = OFDMReceiver(**params)
    rx.padBits = tx.padBits
    rx.SLMPhaseVectorIdx = tx.SLMPhaseVectorIdx
    rbitStream = rx.receive(sig.flatten())
    BER = np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream))
    print("SLM 64 and Clipping : BER = {:<5.2}".format(BER))
    legend.append("OFDM + SLM-64 + Clipping")

    # OFDM with PAPRnet ------------------------------------------
    params['useSLM'] = False
    params['useClipping'] = False
    params['usePAPRnet'] = True
    params['PAPRnetEncoder'] = load_model('./trained_models/PAPRnet01/encoder.hdf5')
    params['PAPRnetDecoder'] = load_model('./trained_models/PAPRnet01/decoder.hdf5')

    tx = OFDMTransmitter(**params)
    sig = tx.transmit(bitStream)
    paprDb.append(calcPAPR(sig))

    rx = OFDMReceiver(**params)
    rx.padBits = tx.padBits
    rbitStream = rx.receive(sig.flatten())
    BER = np.sum(np.logical_xor(rbitStream, bitStream)) / (1.0 * len(bitStream))
    print("PARPnet : BER = {:<5.2}".format(BER))
    legend.append("OFDM + PAPRnet")

    plotCCDF(paprDb, savePath='SLM_vs_PAPRnet.png', show=True, legend=legend,
             title='Complementary Cumulative Distribution Function (CCDF) for PAPR\nComparision between PAPRnet and Conventional techniques')

    print("".join(["-"]*60), "\n\n")

if __name__ == "__main__":

    # L_vs_PAPR()
    # N_vs_PAPR()
    # N_vs_BER()
    # NCP_vs_BER()
    # L_vs_BER()
    # SLM_vs_PAPR()
    # Clipping_vs_PAPR()
    # ClippingSLM_vs_PAPR()
    # ConvCoding_vs_PAPR()
    SLM_vs_NN()
