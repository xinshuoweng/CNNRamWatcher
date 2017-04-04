# Ramwatcher
A very lightweight deep learning tool for monitoring **data flow**, **parameter size** and their corresponding **memory usage** throughout deep neural network. This tool **doesn't** need any powerful computational resource (eg. GPU). And it's very easy to use since it follows many similar rules in popular deep learning frameworks (**Caffe**, **Tensorflow**, **Torch**)

#### Usage:

*1. clone the github repository recursively including xinshuo_toolbox.*
~~~shell
git clone --recursive https://github.com/xinshuoweng/ramwatcher
~~~

*2. install dependency for the toolbox.*
~~~shell
cd ramwatcher/xinshuo_toolbox
pip install -r requirements.txt
~~~

*3. define the network and print the memory info (one might want to look at example.py first for a quick and simple instruction).*
~~~shell
cd ..
python example.py
~~~

#### Example Result:

*Network Info Table:*
<p align="center">
<img alt="Network Info Table", src="network_info.png")
</p>

*Memory Usage Pie Chart:*
<p align="center">
<img alt="Memory Usage Pie Chart", src="memory_chart.png")
</p>

*Model Graph:*
<p align="center">
<img alt="Model Graph", src="model_graph.png")
</p>


#### Limitation:

*1. Only basic layers (Convolution, Pooling, Activation, Dense, Concat) are supported right now. More layers will be added in the future.*<br>
*2. Add prototxt parse function in the future.*<br>
