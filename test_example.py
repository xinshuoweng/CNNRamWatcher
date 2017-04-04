# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import __init__paths__
from layer import Convolution, Pooling, Input, Dense, Activation
from net import gModule



def main():
	# test input layer
	inputlayer1 = Input(name='input', inputshape=(384, 256, 3), datatype='uint')
	model = gModule(inputlayers=inputlayer1)
	conv1 = model.add(Convolution(name='conv1', bottom=inputlayer1, nOutputPlane=64, kernal_size=3, padding=1))
	model.add(Convolution(name='conv2_1', bottom=conv1, nOutputPlane=128, kernal_size=3, padding=1))
	model.add(Convolution(name='conv2_2', bottom=conv1, nOutputPlane=128, kernal_size=3, padding=1))
	model.compile(np.ndarray((16, 384, 256, 3)))

	# model.remove('conv2_2')
	# assert not model.layers.has_key('conv1')
	# assert not model.blobs.has_key('conv1')
	# model.compile(np.ndarray((16, 384, 256, 3)))
	# assert_allclose(model.nb_entries, 3)

	model.summary()

if __name__ == '__main__':
    main()

