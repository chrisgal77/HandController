import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'


class CNNBlock(keras.Model):
    def __init__(self, filters, kernel_size, stride, batch_norm=True):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=stride)
        self.relu = layers.LeakyReLU(alpha=0.1)
        self.use_bn = batch_norm
        if batch_norm:
            self.batch_norm = layers.BatchNormalization()
     
    def call(self, x):
        x = self.batch_norm(self.conv(x)) if self.use_bn else self.conv(x)
        return self.relu(x)
   

class ResidualBlock(keras.Model):
    def __init__(self, channels, use = True, num_reps = 1):
        super(ResidualBlock, self).__init__()
        self.layers = keras.Sequential()
        for _ in range(num_reps):
            self.layers.add(
                keras.Sequential([
                    CNNBlock(channels//2, kernel_size=1, stride=1),
                    CNNBlock(channels, kernel_size=3, stride=1)
                ])
            )
        
        self.use = use
        
    def call(self, x):
        for layer in self.layers.layers:
            x = layer(x) + x if self.use else layer(x)
        
        return x
        

class Scale(keras.Model):
    def __init__(self, channels, num_classes):
        super(Scale, self).__init__()
        self.pred = keras.Sequential([
            CNNBlock(channels, kernel_size=3, stride=1),
            CNNBlock((num_classes+5)*3, batch_norm=False, kernel_size=1, stride=1)
        ])
        self.num_classes = num_classes
        
    def call(self, x):
        return tf.transpose(tf.reshape(self.pred(x), (x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])),
                            perm=[0,1,3,4,2])
    
        
class YOLO(keras.Model):
    def __init__(self, num_classes, architecture):
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
                
            elif  isinstance(element, list):
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
    
    
    
    #yolo = YOLO(7, 2, 20, architecture)
    test = tf.random.normal(shape=(2,448,448,3))
    
    x = yolo(test)
    assert tf.reshape(x, (2, 7, 7, -1)).shape == (2,7,7,30)
    