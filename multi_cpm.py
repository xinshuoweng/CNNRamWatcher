# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import __init__paths__
from layer import Convolution, Pooling, Input, Dense, Activation
from net import Sequential


def stage_network(stage_number):


def main():
	# create network
	inputlayer = Input(name='input', inputshape=(368, 368, 3), datatype='uint')


	network = gModule({})
	head.add(Convolution(name='conv1_1', nOutputPlane=64, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu1_1', function='relu', datatype='single'))
	head.add(Convolution(name='conv1_2', nOutputPlane=64, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu1_2', function='relu', datatype='single'))
	head.add(Pooling(name='pool1', kernal_size=2, stride=2))

	head.add(Convolution(name='conv2_1', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu2_1', function='relu', datatype='single'))
	head.add(Convolution(name='conv2_2', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu2_2', function='relu', datatype='single'))
	head.add(Pooling(name='pool2', kernal_size=2, stride=2))
	
	head.add(Convolution(name='conv3_1', nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu3_1', function='relu', datatype='single'))
	head.add(Convolution(name='conv3_2', nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu3_2', function='relu', datatype='single'))
	head.add(Convolution(name='conv3_3', nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu3_3', function='relu', datatype='single'))
	head.add(Convolution(name='conv3_4', nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu3_4', function='relu', datatype='single'))
	head.add(Pooling(name='pool3', kernal_size=2, stride=2))
	
	head.add(Convolution(name='conv4_1', nOutputPlane=512, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu4_1', function='relu', datatype='single'))
	head.add(Convolution(name='conv4_2', nOutputPlane=512, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu4_2', function='relu', datatype='single'))
	head.add(Convolution(name='conv4_3', nOutputPlane=256, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu4_3', function='relu', datatype='single'))
	head.add(Convolution(name='conv4_4', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	head.add(Activation(name='relu4_4', function='relu', datatype='single'))

	# define stage 1
	stage1_left = Sequential(head)
	# stage1_left.add(Indentity())
	stage1_left.add(Convolution(name='conv5_1_l', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	stage1_left.add(Activation(name='relu5_1_l', function='relu', datatype='single'))
	stage1_left.add(Convolution(name='conv5_2_l', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	stage1_left.add(Activation(name='relu5_2_l', function='relu', datatype='single'))
	stage1_left.add(Convolution(name='conv5_3_l', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	stage1_left.add(Activation(name='relu5_3_l', function='relu', datatype='single'))
	stage1_left.add(Convolution(name='conv5_4_l', nOutputPlane=512, kernal_size=1, padding=0, datatype='single'))
	stage1_left.add(Activation(name='relu5_4_l', function='relu', datatype='single'))
	stage1_left.add(Convolution(name='conv5_5_l', nOutputPlane=38, kernal_size=1, padding=0, datatype='single'))

	stage1_right = Sequential(head)
	stage1_right.add(Indentity())
	stage1_right.add(Convolution(name='conv5_1_r', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	stage1_right.add(Activation(name='relu5_1_r', function='relu', datatype='single'))
	stage1_right.add(Convolution(name='conv5_2_r', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	stage1_right.add(Activation(name='relu5_2_r', function='relu', datatype='single'))
	stage1_right.add(Convolution(name='conv5_3_r', nOutputPlane=128, kernal_size=3, padding=1, datatype='single'))
	stage1_right.add(Activation(name='relu5_3_r', function='relu', datatype='single'))
	stage1_right.add(Convolution(name='conv5_4_r', nOutputPlane=512, kernal_size=1, padding=0, datatype='single'))
	stage1_right.add(Activation(name='relu5_4_r', function='relu', datatype='single'))
	stage1_right.add(Convolution(name='conv5_5_r', nOutputPlane=19, kernal_size=1, padding=0, datatype='single'))

	stage1 = ConcatTable(dim=1)
	stage1.add(head)
	stage1.add(stage1_left)
	stage1.add(stage1_right)




	# network.add(Dense(name='fc1', nOutputPlane=512, datatype='single'))
	# network.add(Activation(name='relu4', function='relu', datatype='single'))
	# network.add(Dense(name='fc2', nOutputPlane=6, datatype='single'))
	# network.add(Activation(name='linear1', function='linear', datatype='single'))
	input_data = np.ndarray((1, 368, 368, 3))

	network.compile(input_data)

	# save network info to a table and also the memory usage chart and model graph to image
	# network_table_save_path = 'table.txt'
	# network_pie_chart_save_path = 'chart.png'
	# network_model_save_path = 'model.pdf'
	# network.summary(model_path=network_model_save_path, table_path=network_table_save_path, chart_path=network_pie_chart_save_path)
	network.summary()

if __name__ == '__main__':
	main()