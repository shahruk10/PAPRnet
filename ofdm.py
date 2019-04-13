import numpy as np
from utils import dec2bin, bin2dec, polar2rect
from modulation import qammod, qamdemod, pskmod, pskdemod
from convcode import Trellis, conv_encode, viterbi_decode

class OFDMTransmitter(object):
    
    def __init__(self, **params):
        # necessary params
        self.N = params['N']
        self.modType = params['modType']
        self.M = params['M']
        # optional params
        self.cyclicPrefix = params['cyclicPrefix'] if 'cyclicPrefix' in params else int(0.25 * self.N)
        self.upsampleFactor = params['upsampleFactor'] if 'upsampleFactor' in params else 1
        self.useConvCode = params['useConvCode'] if 'useConvCode' in params else False
        self.convCodeGMatrix = params['convCodeGMatrix'] if 'convCodeGMatrix' in params else [0o5, 0o7] 
        self.useClipping = params['useClipping'] if 'useClipping' in params else False
        self.clippingPercent = params['clippingPercent'] if 'clippingPercent' in params else 0.75
        self.useSLM = params['useSLM'] if 'useSLM' in params else False 
        self.SLMCandidates = params['SLMCandidates'] if 'SLMCandidates' in params else 16

        self.usePAPRnet = params['usePAPRnet'] if 'usePAPRnet' in params else False
        self.PAPRnetEncoder = params['PAPRnetEncoder'] if 'PAPRnetEncoder' in params else None

        self.dumpModChunks = params['dumpModChunks'] if 'dumpModChunks' in params else False
        
        if self.clippingPercent == 1.0:
            self.useClipping = False
        if self.SLMCandidates == 0:
            self.useSLM = False

        self.SLMPhaseRotationOptions = [1, -1, 1j, -1j]
        self.SLMPhaseVectorIdx = None
        self.seed = 2727

        self.padBits = 0
        self.dividingFactor = int(np.log2(self.M) * self.N / self.upsampleFactor)
        
        if len(self.convCodeGMatrix) == 0:
            self.useConvCode = False

        if self.useConvCode:
            memory = np.array([np.max([len(bin(p))-2 for p in self.convCodeGMatrix])-1])
            self.trellis = Trellis(memory, np.array([self.convCodeGMatrix]))
            self.codeRate = '{}/{}'.format(self.trellis.k, self.trellis.n)
            self.dividingFactor = int(self.dividingFactor * self.trellis.n / self.trellis.k)
        else:
            self.trellis = None
            self.codeRate = None
            
    def transmit(self, bitStream):

        if (len(bitStream) % self.dividingFactor):
            self.padBits = self.dividingFactor - (len(bitStream) % self.dividingFactor)
            txBitStream = np.concatenate([bitStream, np.zeros(self.padBits, dtype=bitStream.dtype)])
        else:
            self.padBits = 0
            txBitStream = bitStream

        # convolutional encoding
        if self.useConvCode:
            nBitsToKeep = int(len(txBitStream) * self.trellis.n  / self.trellis.k)
            txBitStream = conv_encode(txBitStream, self.trellis)
            txBitStream = txBitStream[0:nBitsToKeep]

        # modulation
        # modBitChunks = np.reshape(txBitStream, (-1, int(np.log2(self.M))))
        modSymbols = bin2dec(txBitStream, int(np.log2(self.M)))
        if self.modType == 'qam':
            modSymbolsMapped = qammod(modSymbols, self.M)
        elif self.modType == 'psk':
            modSymbolsMapped = pskmod(modSymbols, self.M)
        else:
            raise("unrecognized modulation type : {}".format(self.modType))
       
        # chunking; each row will be converted to a ofdm symbol
        ofdmChunks = np.reshape(modSymbolsMapped, (-1, int(self.N/self.upsampleFactor)))
        
        # upsamping 
        if self.upsampleFactor > 1:
            allIndexes = np.arange(1, ofdmChunks.shape[-1]+1)
            allIndexes = np.repeat(allIndexes, self.upsampleFactor-1)
            ofdmChunks = np.insert(self.ofdmChunks, allIndexes, 0, axis=-1)

        # selectively mapping to reduce papr
        if self.useSLM:
            ofdmChunks = self.applySLM(ofdmChunks)

        if self.dumpModChunks:
            return ofdmChunks # stored to dump as training data for NN

        if self.usePAPRnet:
            ofdmChunksRect = np.concatenate( [np.expand_dims(ofdmChunks.real, 1), 
                                              np.expand_dims(ofdmChunks.imag, 1)], axis=1)    # splitting real and imaginary comp

            ofdmChunksNN = self.PAPRnetEncoder.predict(ofdmChunksRect)
            ofdmChunks = ofdmChunksNN[:, 0, :] + 1j * ofdmChunksNN[:, 1, :]         # converting to complex vals (a+bj)

        # OFDM
        ofdmSignal = np.fft.ifft(ofdmChunks, n=self.N, axis=-1) # apply ifft on last axis (across columns)  

        # clipping peaks to reduce papr
        if self.useClipping:
            ofdmSignal = self.clipOFDM(ofdmSignal)

        # adding cyclic prefix
        if self.cyclicPrefix > 0:
            ofdmSignal = np.concatenate([ofdmSignal[:, (ofdmSignal.shape[-1]-self.cyclicPrefix) : ], ofdmSignal], axis=-1) 

        return ofdmSignal

    def applySLM(self, ofdmChunks):

        np.random.seed(self.seed)
        self.SLMPhaseVectorIdx  = np.zeros(ofdmChunks.shape[0])
        phaseVecCandidates = np.random.choice(self.SLMPhaseRotationOptions, (self.SLMCandidates, self.N)) 

        slmChunks = np.zeros_like(ofdmChunks)
        for cdx, chunk in enumerate(ofdmChunks):
            
            # multiplying by each phaseVec candidate and finding min obtained papr
            minPAPR = np.inf
            minPdx = None
            for pdx, phaseVec in enumerate(phaseVecCandidates):
                mapped = np.multiply(chunk, phaseVec)
                mappedTimeDomainSq = np.power(np.abs(np.fft.ifft(mapped, self.N, axis=-1)),2) 
                PAPR = np.divide(np.max(mappedTimeDomainSq), np.mean(mappedTimeDomainSq)) 
                if PAPR < minPAPR:
                    minPAPR = PAPR
                    minPdx = pdx

            # storing mapped vec with min papr
            slmChunks[cdx] = np.multiply(chunk, phaseVecCandidates[minPdx])
            self.SLMPhaseVectorIdx [cdx] = minPdx

        return slmChunks

    def clipOFDM(self, ofdmChunks):

        for cdx, chunk in enumerate(ofdmChunks):
            clippingVal = self.clippingPercent  * np.max(np.abs(chunk))
            for vdx, val in enumerate(chunk):
                if np.abs(val) > clippingVal:
                    # clipping magnitude, phase same as before
                    ofdmChunks[cdx, vdx] = polar2rect(clippingVal, np.angle(val, deg=True))

        return ofdmChunks

class OFDMReceiver(object):

    def __init__(self, **params):
        # necessary params
        self.N = params['N']
        self.modType = params['modType']
        self.M = params['M']
        # optional params
        self.cyclicPrefix = params['cyclicPrefix'] if 'cyclicPrefix' in params else int(0.25 * self.N)
        self.upsampleFactor = params['upsampleFactor'] if 'upsampleFactor' in params else 1
        self.useConvCode = params['useConvCode'] if 'useConvCode' in params else False
        self.convCodeGMatrix = params['convCodeGMatrix'] if 'convCodeGMatrix' in params else [0o5, 0o7] 
        self.useSLM = params['useSLM'] if 'useSLM' in params else False 
        self.SLMCandidates = params['SLMCandidates'] if 'SLMCandidates' in params else 16
        
        self.usePAPRnet = params['usePAPRnet'] if 'usePAPRnet' in params else False
        self.PAPRnetDecoder = params['PAPRnetDecoder'] if 'PAPRnetDecoder' in params else None

        if len(self.convCodeGMatrix) == 0:
            self.useConvCode = False

        if self.useConvCode:
            memory = np.array([np.max([len(bin(p))-2 for p in self.convCodeGMatrix])-1])
            self.trellis = Trellis(memory, np.array([self.convCodeGMatrix]))

        if self.SLMCandidates == 0:
            self.useSLM = False

        self.SLMPhaseRotationOptions = [1, -1, 1j, -1j]
        self.SLMPhaseVectorIdx = None
        self.seed = 2727

        self.padBits = 0
        
    def receive(self, signal):

        # creating chunks corresponding to ofdm symbols
        ofdmChunks = np.reshape(signal, (-1, self.N + self.cyclicPrefix))
        # removing cyclic prefix 
        ofdmChunks = ofdmChunks[:, self.cyclicPrefix: ]
        # converting OFDM signal to mapped mod symbols
        modSymbolsMapped = np.fft.fft(ofdmChunks, n=self.N, axis=-1)

        if self.usePAPRnet:
            modSymbolsMappedRect = np.concatenate( [np.expand_dims(modSymbolsMapped.real, 1), 
                                                    np.expand_dims(modSymbolsMapped.imag, 1)], axis=1)    # splitting real and imaginary comp

            modSymbolsMappedNN= self.PAPRnetDecoder.predict(modSymbolsMappedRect)
            modSymbolsMapped = modSymbolsMappedNN[:, 0, :] + 1j * modSymbolsMappedNN[:, 1, :]         # converting to complex vals (a+bj)

        # reversing selective mapping
        if self.useSLM:
            modSymbolsMapped = self.unmapSLM(modSymbolsMapped)

        # downsampling
        if self.upsampleFactor > 1:
            modSymbolsMapped = modSymbolsMapped[:, ::(self.upsampleFactor)]

        # un-mapping mod symbols
        modSymbols = qamdemod(modSymbolsMapped, self.M)

        # converting back to bits
        bitStream = dec2bin(modSymbols.flatten(), int(np.log2(self.M)))

        # convolutional decoding using viterbi algorithm
        if self.useConvCode:
            bitStream = viterbi_decode(bitStream, self.trellis)

        return bitStream[0:len(bitStream)-self.padBits]

    def unmapSLM(self, modSymbols):

        if self.SLMPhaseVectorIdx is None:
            raise Exception("phase vector indexes use for slm not provided")
        elif self.SLMPhaseVectorIdx.shape[0] != modSymbols.shape[0]:
            raise Exception("number of phase vector indexes not the same as number of symbols received")

        np.random.seed(self.seed)
        phaseVecCandidates = np.random.choice(self.SLMPhaseRotationOptions, (self.SLMCandidates, self.N)) 
        
        unmappedSymbols = np.zeros_like(modSymbols)
        for sdx, (pdx, sym) in enumerate(zip(self.SLMPhaseVectorIdx, modSymbols)):
            unmappedSymbols[sdx] = np.multiply(np.conj(phaseVecCandidates[int(pdx)]), sym)   # multiplying by complex conjugate of phase vector 

        return unmappedSymbols
        
