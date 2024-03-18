import numpy as np
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import time

# 加载TFLite模型
interpreter = tflite.Interpreter(model_path='mnist_model_2.tflite')
interpreter.allocate_tensors()

# 获取输入和输出张量的索引
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# 加载保存的测试集
loaded_test_dataset = np.load('test_dataset.npy', allow_pickle=True)

# 进行推理
predictions = []
start_time = time.time()
for sample in loaded_test_dataset:
    # 获取图像和标签
    image, label = np.hsplit(sample, [-1])

    #print("Image shape: ", image.shape)
    #print("Image data type: ", image.dtype)
    input_shape = interpreter.get_input_details()[0]['shape'] 
    #print("Model input shape: ", input_shape)
    image = image.reshape(input_shape).astype(np.float32)
    
    # 设置输入张量
    interpreter.set_tensor(input_index, image.astype(np.float32))
    
    # 进行推理
    interpreter.invoke()
    
    # 获取输出张量
    output_data = interpreter.get_tensor(output_index)
    
    # 预测类别
    predicted_class = np.argmax(output_data)
    
    predictions.append((predicted_class, int(label[0])))

stop_time = time.time()

# 计算准确性
correct = sum(1 for predicted, actual in predictions if predicted == actual)
total = len(loaded_test_dataset)

accuracy = correct / total
print(f'Test total: {total}')
print(f'Test accuracy: {accuracy}')

all_time = stop_time - start_time
print(f'time: {all_time}')
