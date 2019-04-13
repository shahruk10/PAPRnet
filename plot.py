import numpy as np
import matplotlib.pyplot as plt

labelTextSize = 14

def plotBER(BERSets, xval, xlabel, title=None, legend=None, semilog=True, show=True, savePath=None):

    plt.figure(figsize=(16,9))
    for BER in BERSets:
        if semilog:
            BER = np.array(BER)
            BER[BER<1e-3] = -1
            plt.semilogy(xval, BER, '-o', nonposy='mask')
        else:
            plt.plot(xval, BER, '-o')

    plt.tick_params(labelsize=labelTextSize)
    plt.xlabel(xlabel, fontsize=labelTextSize)
    plt.ylabel('BER', fontsize=labelTextSize)
    plt.ylim([0,1.0])
    plt.grid()

    if title is not None:
        plt.title(title, fontsize=labelTextSize)
    else:
        plt.title("Bit Error Rates", fontsize=labelTextSize)

    if legend is not None:
        plt.legend(legend, fontsize=labelTextSize)
    if savePath is not None:
        plt.savefig(savePath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plotCCDF(paprDbSet, steps=0.1, title=None, legend=None,  show=True, savePath=None):

    if not isinstance(paprDbSet, list):
        paprDbList = [paprDbSet]
    else:
        paprDbList = paprDbSet

    plt.figure(figsize=(16, 9))

    # plotting for all sets of paprDb vals provided
    for paprDb in paprDbList:
        valRange = np.arange(np.min(paprDb), np.max(paprDb), steps)
        y = np.zeros(valRange.shape)
        for vdx, val in enumerate(valRange):
            # calculating probability of papr > val
            y[vdx] = (len(paprDb[paprDb > val]) / (1.0 * len(paprDb)))

        # adding to plot
        plt.plot(valRange, y, '-o')

    plt.tick_params(labelsize=labelTextSize)
    plt.xlabel('Z (dB)', fontsize=labelTextSize)
    plt.ylabel('Prob (PAPR > Z)', fontsize=labelTextSize)
    plt.grid()

    if title is not None:
        plt.title(title, fontsize=labelTextSize)
    else:
        plt.title("CCDF for PAPR", fontsize=labelTextSize)

    if legend is not None:
        plt.legend(legend, fontsize=labelTextSize)
    if savePath is not None:
        plt.savefig(savePath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plotImages(imgs):

    plt.figure(figsize=(16,9))
    for i in range(len(imgs)):
        plt.subplot(len(imgs),1,i+1)
        plt.imshow(imgs[i])

    plt.show()