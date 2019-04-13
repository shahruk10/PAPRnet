import numpy as np
from modmaps import mapQAM

def qammod(modSymbols, M, output_type='rect'):
    if M in mapQAM:
        polar = np.array([mapQAM[M][s] for s in modSymbols])
        if output_type == 'rect':
            return polar[:, 0] * np.exp(1j * np.pi * polar[:, 1]/180.0)
        elif output_type == 'polar':
            return polar
        else:
            raise("output_type not recognized : {}".format(output_type))
    else:
        raise("{}-mapQAM not supported yet".format(M))


def qamdemod(modSymbolsMapped, M):

    symbolMap = np.array([mapQAM[M][s] for s in sorted(
        mapQAM[M].keys())])   # getting all symbols
    # converting to rectangular a+bj format
    symbolMap = symbolMap[:, 0] * np.exp(1j * np.pi * symbolMap[:, 1]/180.0)

    rxSymbols = np.expand_dims(modSymbolsMapped, -1)
    rxSymbols = np.repeat(rxSymbols, M, axis=-1)

    # finding euclidean distance
    delta = np.abs(rxSymbols - symbolMap)
    decodedSymbols = np.argmin(delta, axis=-1)

    return decodedSymbols


def pskmod(modSymbols, M, output_type='rect'):
    pass


def pskdemod(modSymbolsMapped, M):
    pass
