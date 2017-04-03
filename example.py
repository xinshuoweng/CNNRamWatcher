# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import __init__paths__
from layer import Convolution, Pooling, Input, Dense, Activation
from net import Sequential



def main():
	# create network
	network = Sequential()
	network.add(Input(name='input', inputshape=(84, 84, 4)))
	network.add(Convolution(name='conv1', nOutputPlane=32, kernal_size=8, stride=4))
	network.add(Activation(name='relu1', function='relu'))
	# network.append(Pooling(name='pool1', kernal_size=2, stride=2))
	network.add(Convolution(name='conv2', nOutputPlane=64, kernal_size=4, stride=2))
	network.add(Activation(name='relu2', function='relu'))
	# network.append(Pooling(name='pool2', kernal_size=2, stride=2))
	network.add(Convolution(name='conv3', nOutputPlane=64, kernal_size=3, stride=1))
	network.add(Activation(name='relu3', function='relu'))
	# network.append(Pooling(name='pool3', kernal_size=2, stride=2))
	network.add(Dense(name='fc1', nOutputPlane=512))
	network.add(Activation(name='relu4', function='relu'))
	network.add(Dense(name='fc2', nOutputPlane=6))
	network.add(Activation(name='linear1', function='linear'))
	input_data = np.ndarray((32, 84, 84, 4))
	network.compile(input_data)

	# save network info to a table and also the memory usage chart and model graph to image
	# network_table_save_path = 'table.txt'
	# network_pie_chart_save_path = 'chart.png'
	# network_model_save_path = 'model.pdf'
	# network.summary(model_path=network_model_save_path, table_path=network_table_save_path, chart_path=network_pie_chart_save_path)
	network.summary()

if __name__ == '__main__':
	main()