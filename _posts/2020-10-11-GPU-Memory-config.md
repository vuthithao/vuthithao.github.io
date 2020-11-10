---
layout: post
title: Config GPU memory
date: 2020-11-10
author: V
header-image: /img/gpu.jpg
catalog: true
tags:
    - tensorflow 
    - machine learning
    - tensorflow lite
    - keras
    - GPU
---

## 1. Tensorflow

### Thêm phần config sau vào Session

- `per_process_gpu_memory_fraction`: %GPU mà bạn muốn sử dụng

```python
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.33
sess = tf.Session(config=config)

```

#### Nếu không biết chính xác bạn cần bao nhiêu mem thay thế config thành đoạn sau:
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
```
Model của bạn sẽ chiếm đủ lượng mem mà nó cần.

## 2. Keras
#### Config cho keras bạn sử dụng cách sau

```python
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, save_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
```