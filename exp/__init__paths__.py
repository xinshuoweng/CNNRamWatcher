# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../xinshuo_toolbox/machine_learning/deep_learning'))
add_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../xinshuo_toolbox/python'))
add_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../xinshuo_toolbox/file_io'))