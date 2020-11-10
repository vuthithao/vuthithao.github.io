---
layout: post
title: Deploy machine learning models trên mobile và thiết bị nhúng
subtitle: Tensorflow model
date: 2020-11-10
author: V
header-image: /img/iPhone12.jpg
catalog: true
tags:
    - tensorflow 
    - machine learning
    - tensorflow lite
    - keras
---

[Colab](https://colab.research.google.com/drive/1Duy_r8bqRVNkvOD6uhywCMeKB6hQCrDU?usp=sharing)
## 1. Framework sử dụng : Tensorflow Lite
## 2. Các bước
- Bước 1: Chọn một model
- Bước 2: Nén Tensorflow model bằng Tensorflow Lite
- Bước 3: Deploy lên mobile hoặc thiết bị nhúng
- Bước 4: Tối ưu 

### Bước 1+2: Chọn model và nén
Các loại models có thể convert sang TFLite: [TF SaveModel](https://www.tensorflow.org/guide/saved_model), [Keras Prebuilt Model](https://www.tensorflow.org/guide/keras/sequential_model), [Concrete Function](https://www.tensorflow.org/guide/intro_to_graphs)
![_config.yml]({{ site.baseurl }}/images/convert.png)

#### 1. TF SaveModel
Tạo một model đơn giản sử dụng Tensorflow và lưu dưới dạng TF SaveModel - bao gồm trọng số và các phép tính toán, không cần build lại model bằng code mà vẫn sử dụng được. Dưới đây là ví dụ convert từ TF SaveModel sang TF Lite FlatBuffer.

```python
# we will train 
import tensorflow as tf

# Construct a basic TF model.
root = tf.train.Checkpoint()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# Save the model into temp directory
export_dir = "/test_saved_model"
input_data = tf.constant(1., shape=[1, 1])
to_save = root.f.get_concrete_function(input_data)
tf.saved_model.save(root, export_dir, to_save)

# Convert the model into TF Lite.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```
##### Load model
```python
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

```
Một ví dụ khác convert và sử dụng MobileNet [tại đây](https://colab.research.google.com/drive/17l1G-9mPjRmEXlAnf0JzBstOgNoDtK6c?usp=sharing)
#### 2. Keras Prebuilt Model
Convert pre-train tf.keras MobileNet sang TF Lite
```python
import numpy as np
import tensorflow as tf

# Load the MobileNet keras model.
# we will create tf.keras model by loading pretrained model on #imagenet dataset
model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(224, 224, 3))
# here we pretrained model no need use SaveModel 
# here we will pass model directly to TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

#if you want to save the TF Lite model use below steps or else skip
# Save the model.
with open('kerasmodel.tflite', 'wb') as f:
  f.write(tflite_model)
```
##### Load model giống như ở 1
#### 3. Concrete Function
Với tensorflow 2.0 model còn được lưu dưới dạng Concrete Function.

###### Ví dụ với keras MobileNet model
```python
import tensorflow as tf
# load mobilenet model of keras 
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))
```
```python
#get callable graph from model. 
run_model = tf.function(lambda x: model(x))
# to get the concrete function from callable graph 
concrete_funct = run_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

#convert concrete function into TF Lite model using TFLiteConverter
converter =  tf.lite.TFLiteConverter.from_concrete_functions([concrete_funct])
tflite_model = converter.convert()
# Save the model.
with open('tf2model.tflite', 'wb') as f:
  f.write(tflite_model)
```
##### Load model giống như ở 1

#### 4. Sử dụng command line
```bash
tflite_convert --post_training_quantize  --output_file facenet.tflite --keras_model_file model/facenet_keras.h5
```
- `--post_training_quantize`: có sử sụng quantize để optimize
- `--output_file`: TF Lite model
- `--keras_model_file`: keras model (h5)

```bash
tflite_convert --saved_model_dir mobilenet_saved_model --output_file mobilenet.tflite
```
- `--output_file`: TF Lite model
- `--saved_model_dir`: TF SaveModel
### Bước 4: Optimize sử dụng lượng tử hóa (quantization) (optional)

Lượng tử hóa, từ full floating point sang float16 hoặc 8-bit integers

Thử với model trong phần `1` của `Bước 1+2`
```python
import tensorflow as tf

saved_model_dir = 'model.tflite'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_quantized_model)
```

- Trong thực tế, nhiều TF model được lưu dưới dạng checkpoints, chúng ta cần đưa về dạng TF SaveModel, hoặc keras trước khi convert sang TF Lite
- Một ví dụ chuyển đổi FaceNet từ TF model sang TF Lite model [tại đây](https://colab.research.google.com/drive/1VovEl0I671JG7ufg2PtfjwKdM8YEK353?usp=sharing) 
- Sử dụng FaceNet TF Lite vào project phân loại [tại đây](http://gitlab.giaingay.io/vuthithao/face-classification)
- Yêu cầu quyền truy cập liên hệ `vuthithao04081996@gmail.com`
