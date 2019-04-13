import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, ReLU, Lambda, Concatenate, Flatten, Add

def IFFT(sig, name=None):
    N = int(sig.shape[-1])
    return Lambda(tf.ifft, name=name, output_shape=(1,N))(sig)

def FloatToComplex(sig, name=None):
    N = int(sig.shape[-1])
    return Lambda(lambda x : tf.complex(x[:, 0:1, :], x[:, 1:2, :]), name=name, output_shape=(1,N))(sig)

def ComplexToFloat(sig, name=None):
    N = int(sig.shape[-1])
    return Concatenate(name=name, axis=-2)([Lambda(tf.real, output_shape=(1, N), name=name+'_Re')(sig), Lambda(tf.imag, output_shape=(1, N), name=name+'_Im')(sig)])

def PAPRnetEncoder(N):

    enc_in = Input((2, N), name="encoder_input")

    h1 = Dense(N*2, activation='relu', name='Dense1')(enc_in)
    h1 = BatchNormalization(name='DenseBN1')(h1)

    h2 = Dense(N*2, activation='relu', name='Dense2')(h1)
    h2 = BatchNormalization(name='DenseBN2')(h2)

    h3 = Dense(N, activation='relu', name='Dense3')(h2)
    h3 = BatchNormalization(name='DenseBN3')(h3)

    enc_out = Lambda(lambda x: x, name='encoder_output')(h3)

    return Model(inputs=[enc_in], outputs=[enc_out], name="PAPRnet_Encoder")

def PAPRnetDecoder(N):

    dec_in = Input((2,N), name="decoder_input")

    h4 = Dense(N*2, activation='relu', name='Dense4')(dec_in)
    h4 = BatchNormalization(name='DenseBN4')(h4)

    h5 = Dense(N*2, activation='relu', name='Dense5')(h4)
    h5 = BatchNormalization(name='DenseBN5')(h5)

    dec_out = Dense(N, activation='linear', name='decoder_output')(h5)

    return Model(inputs=[dec_in], outputs=[dec_out], name="PAPRnet_Decoder")

def PAPRnetAutoEncoder(N, enc, dec):
    
    # auto encoder
    enc_in = enc.input
    enc_out = enc(enc_in)
    dec_out = dec(enc_out)

    # taking ifft of encoder output - used to minimize PAPR
    cmplx = FloatToComplex(enc_out, name='EncoderOut-FloatToComplex')
    ifft = IFFT(cmplx, name='%d-IFFT' % N)
    ifft = ComplexToFloat(ifft, name='%d-IFFT-ComplexToFloat' % N)

    return Model(inputs=[enc_in], outputs=[dec_out, ifft])
