"""   JLL, 2021.11.3 ToDo
from /home/jinn/YPN/OPNet/train_modelB3.py
train modelA4 = UNet on comma10k data

1. Use OP supercombo I/O
2. Task: Multiclass semantic segmentation (NUM_CLASSES = 6)
3. Loss: tf.keras.losses.CategoricalCrossentropy
4. Input:
   X_batch.shape = (2, 2x6, 128, 256) (num_channels = 6, 2 yuv images)
   Y_batch.shape = (2, 256, 512, 2x6) (num_classes  = 6)
5. Output: Loss

Run:
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverA4.py
   (YPN) jinn@Liu:~/YPN/OPNet$ python train_modelA4.py
Input:
   /home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5  (X for debugging)
   /home/jinn/dataAll/comma10k/Xmasks/*.png
Output:
   /OPNet/saved_model/modelA4_loss.npy
   plt.title("Training Loss")
   plt.title("Training Accuracy")
   plt.title("Validation Loss")
   plt.title("Validation Accuracy")
   plot_predictions(train_images[:4], colormap, model=model)
     binary mask: one-hot encoded tensor = (?, ?, ?)
     visualize: RGB segmentation masks (each pixel by a unique color corresponding
       to each predicted label from the human_colormap.mat file)
"""
import os
import h5py
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from modelA4 import get_model
from serverA4 import client_generator

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y = tup
      #--- Y.shape = (25, 2, 56)
    X_batch = X
    Y_batch = Y.reshape(Y.shape[0], -1)
    print('#--- X_batch.shape =', X_batch.shape)
    print('#--- Y_batch.shape =', Y_batch.shape)
    #print('#--- Y.shape[0] =', Y.shape[0])

      #--- X_batch.shape = (25, 12, 128, 256)
      #--- Y_batch.shape = (25, 112)
    yield X_batch, Y_batch

if __name__=="__main__":
  start_time = time.time()
  parser = argparse.ArgumentParser(description='Training modelA4')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  #parser.add_argument('--epoch', type=int, default=30, help='Number of epochs.')
  parser.add_argument('--epoch', type=int, default=1, help='Number of epochs.')
  #parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--epochsize', type=int, default=1, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')

  args = parser.parse_args()

  # Build model
  img_shape = (12, 128, 256)
  num_classes = 3
  model = get_model(img_shape, num_classes)
  model.summary()

'''
  filepath = "./saved_model/opEffPNb3-BestWeights.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min')
  callbacks_list = [checkpoint]

  adam = tf.keras.optimizers.Adam(lr=0.0001)
  #model.load_weights('./saved_model/opEffPNb3-BestWeights.hdf5', by_name=True)
  model.compile(optimizer=adam, loss="mse")

  history = model.fit(
    gen(20, args.host, port=args.port),
    steps_per_epoch=1, epochs=5,
    validation_steps=20*1200/25., verbose=1, callbacks=callbacks_list)

  print('#--- # of epochs are run =', len(history.history['loss']))
  model.save('./saved_model/opUNetPNB3.h5')

  np.save('./saved_model/opUNetPNB3_loss', np.array(history.history['loss']))
  lossnpy = np.load('./saved_model/opUNetPNB3_loss.npy')
  plt.plot(lossnpy)
  plt.draw() #plt.show()
  print('#--- modelA4 lossnpy.shape =', lossnpy.shape)
  plt.pause(0.5)
  input("Press ENTER to exit ...")
  plt.close()
'''
