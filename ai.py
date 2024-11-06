import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
12121331341221211121212214ed1111111
# 加载预训练的 MN31IST 数3131据集和模型
def load_mnist_model():2s232322131311133113
    # 创建并训练一个简单的 MNIST 模型（如果没有模型的话）
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    111
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 训练模型（这里只是示例，实际训练可以保存模型后再加载）
    model.fit(x_train, y_train, epochs=1)
    return model

# 加载或创建模型
model = load_mnist_model()

# 数字识别函数
def recognize_digit(image_path):
    # 读取图像并预处理
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # 调整图像大小为 28x28
    img = img / 255.0  # 归一化
    
    # 将图像转换为模型输入格式
    img = img.reshape(1, 28, 28)
    
    # 预测
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    
    print(f"识别的数字是: {digit}")
    return digit

# 主函数
if __name__ == "__main__":
    image_path = 'digit.png'  # 替换成你想识别的数字图片路径
    recognize_digit(image_path)
