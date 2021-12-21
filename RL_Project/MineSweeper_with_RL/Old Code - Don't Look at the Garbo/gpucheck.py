# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:05:32 2021

@author: joshc
"""

import tensorflow as tf
print(tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Hello World")