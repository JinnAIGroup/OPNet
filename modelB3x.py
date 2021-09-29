'''   JLL, 2021.9.18-20
Build modelB3 = Tiny EfficientNet (Eff) + Pose Net (PN) = opEffPNb3

1. Use OP's supercombo079.html (Leon's) I/O
   https://drive.google.com/file/d/1L8sWgYKtH77K6Kr3FQMETtAWeQNyyb8R/view
   file:///home/jinn/snpe/dlc/supercombo079.html
2. Read carefully Leon's main.py, camera.py, etc. for OP I/O format (very complicated)
3. Input shape = (None, 12, 128, 256)
4. Output shape =  (None, 2383)   # ToDo
5. Run1: (YPN) jinn@Liu:~/YPN/Leon$ python modelB3.py
'''

import tensorflow as tf
from tensorflow.keras import layers

def Eff(inputs):
  x = layers.SeparableConv2D(32, 3, strides=(2,2), padding="same", activation="elu")(inputs)
  x = layers.SeparableConv2D(64, 3, strides=(2,2), padding="same", activation="elu")(x)
  x = layers.SeparableConv2D(128, 3, strides=(2,2), padding="same", activation="elu")(x)
  x = layers.SeparableConv2D(256, 3, strides=(2,2), padding="same", activation="elu")(x)
  x = layers.SeparableConv2D(512, 3, strides=(2,2), padding="same", activation="elu")(x)
  x = layers.Conv2D(1024, 1, padding="same", activation="elu")(x)
  x = layers.Conv2D(32, 1, padding="same", activation="elu")(x)
  x = layers.Flatten()(x)
  return x

def RNN(x):
  return x

def PN(x):
  x1 = layers.Dense(64, activation='relu')(x)
  x2 = layers.Dense(64, activation='relu')(x)
  x3 = layers.Dense(64, activation='relu')(x)
  x1 = layers.Dense(51)(x1)
  x2 = layers.Dense(50)(x2)
  x3 = layers.Dense(5)(x3)
  x = layers.Concatenate(axis=-1)([x1, x2, x3])
  #---  x1.shape, x2.shape, x3.shape = (None, 51) (None, 50) (None, 5)
  #--- in modelgen.py, output1, output3, output2 = (None, 51) (None, 50) (None, 5)
  return x

def path(x):
  return x

def left_lane(x):
  return x

def right_lane(x):
  return x

def long_x(x):
  return x

def long_v(x):
  return x

def long_a(x):
  return x

def lead(x):
  return x

def desire_state(x):
  return x

def desire_pred(x):
  return x

def pose(x):
  return x

def meta(x):
  return x

def call(inputs):
  x = Eff(inputs)
  x = PN(x)
  return x

if __name__=="__main__":
  input1 = tf.keras.Input(shape=(12, 128, 256), name='img')
  output = call(input1)
  model = tf.keras.Model(inputs=[input1], outputs=output, name="modelB3")
  adam = tf.keras.optimizers.Adam(lr=0.0001)
  model.compile(optimizer=adam)
  model.summary()
