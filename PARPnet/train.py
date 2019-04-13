
import sys
import glob
import numpy as np

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

from PAPRnet import PAPRnetEncoder, PAPRnetDecoder, PAPRnetAutoEncoder
from customLoss import paprLoss


N = 512

x = np.load('data/modChunks.npz')['data']
paprTargets = np.ones((x.shape[0],1), dtype=np.float32) * 3.0
x = np.concatenate( [np.expand_dims(x.real, 1), np.expand_dims(x.imag, 1)], axis=1)
y = np.zeros(x.shape[0], dtype=np.float32)

xtrain = x[1000:]
ytrain = y[1000:]
xval = x[:1000]
yval = x[:1000]

encoder = PAPRnetEncoder(N)
decoder = PAPRnetDecoder(N)
autoencoder = PAPRnetAutoEncoder(N, encoder, decoder)

plot_model(autoencoder, to_file='./autoencoder.png')
print(autoencoder.summary())

autoencoder.compile(loss=['mse', paprLoss], loss_weights=[1.0,0.01], optimizer='adam')

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.01, cooldown=0, min_lr=1e-10))
callbacks.append(CSVLogger('trainingLog.csv', separator=',', append=False))
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=0, mode='auto'))

autoencoder.fit(xtrain, [xtrain, ytrain], validation_data=(xval, [xval,yval]), batch_size=512, epochs=100, callbacks=callbacks)
autoencoder.save('autoencoder.hdf5')
encoder.save('encoder.hdf5')
decoder.save('decoder.hdf5')