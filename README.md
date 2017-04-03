# CNN Memory Usage Monitor
A simple deep learning tool for monitoring data and parameter usage throughout CNN.

#### Usage:

*1. clone the github repository recursively including xinshuo_toolbox.*
~~~shell
git clone --recursive https://github.com/xinshuoweng/cnn_memory_usage_monitor
~~~

*2. install dependency for the toolbox.*
~~~shell
cd cnn_memory_usage_monitor
pip install requirements.txt
~~~

*3. define the network and print the memory info (one might want to look at example.py first for a quick and simple instruction).*
~~~shell
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

*1. Only sequential model is supported right now. More operation for constructing the graph will be added depended on demand*
*2. Only basic layers (Convolution, Pooling, Activation, Dense) are supported right now. More layers will be added in the future.*
