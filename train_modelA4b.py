"""   JLL, 2021.11.5, 11.9
from /home/jinn/YPN/OPNet/train_modelA4a.py
train modelA4 = UNet on comma10k data
use custom_loss

1. Use OP supercombo I/O
2. Task: Multiclass semantic segmentation (num_classes = 6)
3. Loss: tf.keras.losses.CategoricalCrossentropy

Run: on 3 terminals
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverA4.py --port 5557
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverA4.py --port 5558 --validation
   (YPN) jinn@Liu:~/YPN/OPNet$ python train_modelA4.py --port 5557 --port_val 5558
Input:
   X_img:   /home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5  (X for debugging with 10 imgs)
   Y_GTmsk: /home/jinn/dataAll/comma10k/Xmasks/*.png
   X_batch.shape = (none, 2x6, 128, 256) (num_channels = 6, 2 yuv images)
   Y_batch.shape = (none, 256, 512, 2x6) (num_classes  = 6)
Output:
   /OPNet/saved_model/modelA4_loss.npy
   plt.title("Training Loss")
   plt.title("Training Accuracy")
   plt.title("Validation Loss")
   plt.title("Validation Accuracy")
   plot_predictions(train_images[:4], colormap, model=model)
     binary mask: one-hot encoded tensor = (None, 256, 512, 2x6)
     visualize: RGB segmentation masks (each pixel by a unique color corresponding
       to each predicted label from the human_colormap.mat file)

Data Sets:
  serverA4: train_len = int(0.6*len(all_images)): 6/10
  DeepLabV3+: 1000/(1000+50),  BATCH_SIZE = 4, EPOCHS = 25
  comma10k: BATCH_SIZE = 28, 7, EPOCHS = 100, 30
    self.train_dataset ... if not x.endswith('9.png')
    self.valid_dataset ... if x.endswith('9.png')
    Yousfi 0.044 validation loss; commaai 0.051

Training History:
  #---datagenA4  imgsN = 8
  #---datagenA4  imgsN = 2
  EPOCHS = 20: loss: 1.5808 - accuracy: 0.3710 - val_loss: 1.7751 - val_accuracy: 0.1851
  EPOCHS = 40: loss: 1.4930 - accuracy: 0.4344 - val_loss: 1.7693 - val_accuracy: 0.1843

  model.load_weights

  loss: 1.4894 - accuracy: 0.4396 - val_loss: 1.7646 - val_accuracy: 0.2514
  loss: 1.4646 - accuracy: 0.4918 - val_loss: 1.7503 - val_accuracy: 0.2697

  #---datagenA4  imgsN = 180
  #---datagenA4  imgsN = 20
  EPOCHS = 10 + 10 + 10
  BATCH_SIZE = 8
  loss: 1.4313 - accuracy: 0.5135 - val_loss: 1.5886 - val_accuracy: 0.3667
  Training Time: 00:09:37.29
  loss: 1.4202 - accuracy: 0.5198 - val_loss: 1.4471 - val_accuracy: 0.5059
  Training Time: 00:09:30.82
  loss: 1.4117 - accuracy: 0.4779 - val_loss: 1.4259 - val_accuracy: 0.5112
  Training Time: 00:09:54.38 (2021.11.9 14:16)

  #---datagenA4  imgsN = 8710
  #---datagenA4  imgsN = 967
  EPOCHS = 10
  BATCH_SIZE = 8
  Epoch 1/10
  loss: 1.4339 - accuracy: 0.4489 - val_loss: 1.4633 - val_accuracy: 0.3993
  Epoch 2/10
  loss: 1.4295 - accuracy: 0.4537 - val_loss: 1.4611 - val_accuracy: 0.4620
  Epoch 3/10
  loss: 1.4266 - accuracy: 0.4564 - val_loss: 1.4589 - val_accuracy: 0.4277
  Epoch 4/10
  loss: 1.4252 - accuracy: 0.4582 - val_loss: 1.4606 - val_accuracy: 0.3853
  Epoch 5/10
  loss: 1.4244 - accuracy: 0.4588 - val_loss: 1.4616 - val_accuracy: 0.3997
  Epoch 6/10
  loss: 1.4235 - accuracy: 0.4599 - val_loss: 1.4594 - val_accuracy: 0.4562
  Epoch 7/10
  loss: 1.4223 - accuracy: 0.4618 - val_loss: 1.4583 - val_accuracy: 0.4256
  Epoch 8/10
  loss: 1.4217 - accuracy: 0.4631 - val_loss: 1.4580 - val_accuracy: 0.3789
  Epoch 9/10
  loss: 1.4217 - accuracy: 0.4634 - val_loss: 1.4572 - val_accuracy: 0.4124
  Epoch 10/10
  loss: 1.4206 - accuracy: 0.4651 - val_loss: 1.4587 - val_accuracy: 0.4232
  Training Time: 08:40:12.15

  (convergence too slow: Yousfi used Cosine Annealing)
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
from serverA4 import client_generator, train_len, valid_len, BATCH_SIZE

EPOCHS = 10

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def gen(hwm, host, port, model):
    for tup in client_generator(hwm=hwm, host=host, port=port):
        X_batch, Y_batch = tup
        #print('#---  X_batch[0,   :,  64, 128] =', X_batch[0, :, 64,  128])
        #print('#---  Y_batch[0, 128, 256,   :] =', Y_batch[0, 128, 256, :])

        Y_pred = model.predict(x=X_batch)
        loss = custom_loss(Y_batch, Y_pred)

          #--- X_batch.shape = (2, 12, 128, 256)
          #--- Y_batch.shape = (2, 256, 512, 12)
        yield X_batch, Y_batch

def custom_loss(y_true, y_pred):
      #---  y_true.shape = (None, None, None, None)???
      #---  y_pred.shape = (None???, 256, 512, 12)
      #---  y_true.shape = (2, 256, 512, 12)
      #---  y_pred.shape = (2, 256, 512, 12)
    #print('#---  y_true[0, 128, 256, :] =', y_true[0, 128, 256, :])
    #print('#---  y_pred[0, 128, 256, :] =', y_pred[0, 128, 256, :])
      #---  y_true[0,  64, 128, :] = Tensor("custom_loss/strided_slice:0", shape=(None,), dtype=float32)???
      #---  y_true[0,  64, 128, :] = Tensor("custom_loss/strided_slice_1:0", shape=(None,), dtype=float32)???
      # ??? problems are solved by loss = custom_loss(Y_batch, p)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss12 = cce(y_true, y_pred)   # y_true.shape = (2, 256, 512, 12)
      # NumPy Array Slicing: https://www.w3schools.com/python/numpy/trypython.asp?filename=demo_numpy_array_slicing_2d
    loss1 = cce(y_true[:, :, :,  0:6], y_pred[:, :, :,  0:6])
    loss2 = cce(y_true[:, :, :, 6:12], y_pred[:, :, :, 6:12])
    loss = (loss1 + loss2)/2
    #print('#---  loss12 =', loss12)
    #print('#---  loss1 =', loss1)
    #print('#---  loss2 =', loss2)
    #print('#---  loss =', loss)
      #---  loss = Tensor("custom_loss/categorical_crossentropy/weighted_loss/value:0", shape=(), dtype=float32)
      #---  loss = tf.Tensor(4.938825, shape=(), dtype=float32)

    return loss

if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Training modelA4')
    parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    parser.add_argument('--port_val', type=int, default=5556, help='Port of server for validation dataset.')
    args = parser.parse_args()

    print('#---  train_len, valid_len =', train_len, valid_len)
    # Build model
    img_shape = (12, 128, 256)
    num_classes = 6
    model = get_model(img_shape, num_classes)
    #model.summary()

    filepath = "./saved_model/modelA4-BestWeights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.load_weights('./saved_model/modelA4-BestWeights.hdf5', by_name=True)
    model.compile(optimizer=adam, loss=custom_loss, metrics=["accuracy"])

      # https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    history = model.fit(
      gen(20, args.host, port=args.port, model=model),
      steps_per_epoch=train_len//BATCH_SIZE, epochs=EPOCHS,
      validation_data=gen(20, args.host, port=args.port_val, model=model),
      validation_steps=valid_len//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
        # steps_per_epoch = Total Training Samples / Training Batch Size
        # validation_steps = total_validation_samples / validation_batch_size

    model.save('./saved_model/modelA4.h5')   # 35.6 M

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    np.save('./saved_model/modelA4_loss', np.array(history.history['loss']))
    lossnpy = np.load('./saved_model/modelA4_loss.npy')
    plt.plot(lossnpy)
    plt.title("Training Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.draw() #plt.show()
    #print('#---  modelA4 lossnpy.shape =', lossnpy.shape)
    plt.pause(0.5)
    input("Press ENTER to exit ...")
    plt.close()
'''
'''
