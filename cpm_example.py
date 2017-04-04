# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import __init__paths__
from layer import *
from net import gModule


def stage_network(model, conv4_4, concat_stage, stage_number):
	# define stage 2-5
	conv1_stage_l = model.add(Convolution(name='conv1_stage%d_l' % stage_number, bottom=concat_stage, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv2_stage_l = model.add(Convolution(name='conv2_stage%d_l' % stage_number, bottom=conv1_stage_l, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv3_stage_l = model.add(Convolution(name='conv3_stage%d_l' % stage_number, bottom=conv2_stage_l, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv4_stage_l = model.add(Convolution(name='conv4_stage%d_l' % stage_number, bottom=conv3_stage_l, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv5_stage_l = model.add(Convolution(name='conv5_stage%d_l' % stage_number, bottom=conv4_stage_l, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv6_stage_l = model.add(Convolution(name='conv6_stage%d_l' % stage_number, bottom=conv5_stage_l, nOutputPlane=128, kernal_size=1, padding=0, datatype='single'))
	conv7_stage_l = model.add(Convolution(name='conv7_stage%d_l' % stage_number, bottom=conv6_stage_l, nOutputPlane=38, kernal_size=1, padding=0, datatype='single'))

	conv1_stage_r = model.add(Convolution(name='conv1_stage%d_r' % stage_number, bottom=concat_stage, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv2_stage_r = model.add(Convolution(name='conv2_stage%d_r' % stage_number, bottom=conv1_stage_r, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv3_stage_r = model.add(Convolution(name='conv3_stage%d_r' % stage_number, bottom=conv2_stage_r, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv4_stage_r = model.add(Convolution(name='conv4_stage%d_r' % stage_number, bottom=conv3_stage_r, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv5_stage_r = model.add(Convolution(name='conv5_stage%d_r' % stage_number, bottom=conv4_stage_r, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv6_stage_r = model.add(Convolution(name='conv6_stage%d_r' % stage_number, bottom=conv5_stage_r, nOutputPlane=128, kernal_size=1, padding=0, datatype='single'))
	conv7_stage_r = model.add(Convolution(name='conv7_stage%d_r' % stage_number, bottom=conv6_stage_r, nOutputPlane=19, kernal_size=1, padding=0, datatype='single'))

	concat_stage_new = model.add(Concat(name='concat_stage%d' % (stage_number + 1), bottom=[conv4_4, conv7_stage_l, conv7_stage_r], axis=2))
	return concat_stage_new

def main():
	# create network
	inputlayer = Input(name='input', inputshape=(368, 368, 3), datatype='uint')
	model = gModule(inputlayer)

	# define the head
	conv1_1 = model.add(Convolution(name='conv1_1', bottom=inputlayer, nOutputPlane=64, kernal_size=3, padding=1, datatype='single'))
	conv1_2 = model.add(Convolution(name='conv1_2', bottom=conv1_1, nOutputPlane=64, kernal_size=3, padding=1, datatype='single'))
	pool1 = model.add(Pooling(name='pool1', bottom=conv1_2, kernal_size=2, stride=2))

	conv2_1 = model.add(Convolution(name='conv2_1', bottom=pool1, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	conv2_2 = model.add(Convolution(name='conv2_2', bottom=conv2_1, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	pool2 = model.add(Pooling(name='pool2', bottom=conv2_2, kernal_size=2, stride=2))
	
	conv3_1 = model.add(Convolution(name='conv3_1', bottom=pool2, nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	conv3_2 = model.add(Convolution(name='conv3_2', bottom=conv3_1, nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	conv3_3 = model.add(Convolution(name='conv3_3', bottom=conv3_2, nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	conv3_4 = model.add(Convolution(name='conv3_4', bottom=conv3_3, nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	pool3 = model.add(Pooling(name='pool3', bottom=conv3_4, kernal_size=2, stride=2))
	
	conv4_1 = model.add(Convolution(name='conv4_1', bottom=pool3, nOutputPlane=512, kernal_size=3, padding=1, datatype='single'))
	conv4_2 = model.add(Convolution(name='conv4_2', bottom=conv4_1, nOutputPlane=512, kernal_size=3, padding=1, datatype='single'))
	conv4_3 = model.add(Convolution(name='conv4_3', bottom=conv4_2, nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	conv4_4 = model.add(Convolution(name='conv4_4', bottom=conv4_3, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))

	# define stage 1
	conv5_1_l = model.add(Convolution(name='conv5_1_l', bottom=conv4_4, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	conv5_2_l = model.add(Convolution(name='conv5_2_l', bottom=conv5_1_l, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	conv5_3_l = model.add(Convolution(name='conv5_3_l', bottom=conv5_2_l, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	conv5_4_l = model.add(Convolution(name='conv5_4_l', bottom=conv5_3_l, nOutputPlane=512, kernal_size=1, padding=0, datatype='single'))
	conv5_5_l = model.add(Convolution(name='conv5_5_l', bottom=conv5_4_l, nOutputPlane=38, kernal_size=1, padding=0, datatype='single'))

	conv5_1_r = model.add(Convolution(name='conv5_1_r', bottom=conv4_4, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	conv5_2_r = model.add(Convolution(name='conv5_2_r', bottom=conv5_1_r, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	conv5_3_r = model.add(Convolution(name='conv5_3_r', bottom=conv5_2_r, nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	conv5_4_r = model.add(Convolution(name='conv5_4_r', bottom=conv5_3_r, nOutputPlane=512, kernal_size=1, padding=0, datatype='single'))
	conv5_5_r = model.add(Convolution(name='conv5_5_r', bottom=conv5_4_r, nOutputPlane=19, kernal_size=1, padding=0, datatype='single'))

	concat_stage2 = model.add(Concat(name='concat_stage2', bottom=[conv4_4, conv5_5_l, conv5_5_r], axis=2))

	concat_stage3 = stage_network(model, conv4_4, concat_stage2, 2)
	concat_stage4 = stage_network(model, conv4_4, concat_stage3, 3)
	concat_stage5 = stage_network(model, conv4_4, concat_stage4, 4)
	concat_stage6 = stage_network(model, conv4_4, concat_stage5, 5)

	# define stage 6 (output stage)
	stage_number = 6
	conv1_stage6_l = model.add(Convolution(name='conv1_stage%d_l' % stage_number, bottom=concat_stage6, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv2_stage6_l = model.add(Convolution(name='conv2_stage%d_l' % stage_number, bottom=conv1_stage6_l, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv3_stage6_l = model.add(Convolution(name='conv3_stage%d_l' % stage_number, bottom=conv2_stage6_l, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv4_stage6_l = model.add(Convolution(name='conv4_stage%d_l' % stage_number, bottom=conv3_stage6_l, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv5_stage6_l = model.add(Convolution(name='conv5_stage%d_l' % stage_number, bottom=conv4_stage6_l, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv6_stage6_l = model.add(Convolution(name='conv6_stage%d_l' % stage_number, bottom=conv5_stage6_l, nOutputPlane=128, kernal_size=1, padding=0, datatype='single'))
	conv7_stage6_l = model.add(Convolution(name='conv7_stage%d_l' % stage_number, bottom=conv6_stage6_l, nOutputPlane=38, kernal_size=1, padding=0, datatype='single'))

	conv1_stage6_r = model.add(Convolution(name='conv1_stage%d_r' % stage_number, bottom=concat_stage6, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv2_stage6_r = model.add(Convolution(name='conv2_stage%d_r' % stage_number, bottom=conv1_stage6_r, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv3_stage6_r = model.add(Convolution(name='conv3_stage%d_r' % stage_number, bottom=conv2_stage6_r, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv4_stage6_r = model.add(Convolution(name='conv4_stage%d_r' % stage_number, bottom=conv3_stage6_r, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv5_stage6_r = model.add(Convolution(name='conv5_stage%d_r' % stage_number, bottom=conv4_stage6_r, nOutputPlane=128, kernal_size=7, padding=3, datatype='single'))
	conv6_stage6_r = model.add(Convolution(name='conv6_stage%d_r' % stage_number, bottom=conv5_stage6_r, nOutputPlane=128, kernal_size=1, padding=0, datatype='single'))
	conv7_stage6_r = model.add(Convolution(name='conv7_stage%d_r' % stage_number, bottom=conv6_stage6_r, nOutputPlane=19, kernal_size=1, padding=0, datatype='single'))

	input_data = np.ndarray((1, 368, 368, 3))
	model.compile(input_data)
	model.summary()

if __name__ == '__main__':
	main()