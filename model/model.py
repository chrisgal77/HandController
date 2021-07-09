import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'


architecture = [
    (7, 64, 2, 3),
    'maxpool',
    (3, 192, 1, 1),
    'maxpool',
    (1, 128, 1, 0),
    (3, 192, 1, 1),
    (3, 192, 1, 0),
    (3, 192, 1, 1),
    'maxpool',
    [4, (1, 256, 1, 0), (3, 512, 1, 1)],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'maxpool',
    [2, (1, 512, 1, 0), (3, 1024, 1, 1)],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(keras.Model):
    def __init__(self, kernel_size, filters, stride, padding):
        super(CNNBlock, self).__init__()
        self.padding = layers.ZeroPadding2D(padding=padding)
        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=stride)
        self.relu = layers.LeakyReLU(alpha=0.1)
        self.batch_norm = layers.BatchNormalization()
     
    def call(self, x, training=False):
        x = self.padding(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)
   
        
class YOLO(keras.Model):
    def __init__(self, split_size, num_boxes, num_classes, architecture):
        super(YOLO, self).__init__()
        self._build_network(architecture)
        self.classifier = keras.Sequential([
            layers.Flatten(),
            layers.Dense(512),
            layers.Dropout(0.5),
            layers.LeakyReLU(alpha=0.1),
            layers.Dense(split_size * split_size * (5* num_boxes + num_classes))]
        )
    
    def call(self, x):
        x = self.convs(x)
        return self.classifier(x)
    
    def _build_network(self, architecture):
        self.convs = keras.Sequential()
        
        for element in architecture:
            if isinstance(element, str):
                self.convs.add(layers.MaxPooling2D(pool_size=2, strides=2))
                
            elif isinstance(element, list):
                n_times = element[0]
                kernel_size1, filters1, stride1, padding1 = element[1]
                kernel_size2, filters2, stride2, padding2 = element[2]
                for _ in range(n_times):
                    self.convs.add(CNNBlock(kernel_size1, filters1, stride1, padding1))
                    self.convs.add(CNNBlock(kernel_size2, filters2, stride2, padding2))
                    
            else:
                kernel_size, filters, stride, padding = element
                self.convs.add(CNNBlock(kernel_size, filters, stride, padding))             


def get_model():
    pass



if __name__ == '__main__':
    
    test = tf.random.normal(shape=(3,122,122,3))
    conv = CNNBlock(3,16,2,2)
    x = conv(test)
    print('DONE')
    
    yolo = YOLO(7, 2, 20, architecture)
    test = tf.random.normal(shape=(2,448,448,3))
    
    x = yolo(test)
    assert tf.reshape(x, (2, 7, 7, -1)).shape == (2,7,7,30)
    