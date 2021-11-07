"""   JLL, 2021.11.5, 11.7
from /home/jinn/YPN/OPNet/train_modelB3.py
train modelA4 = UNet on comma10k data

1. Use OP supercombo I/O
2. Task: Multiclass semantic segmentation (num_classes = 6)
3. Loss: tf.keras.losses.CategoricalCrossentropy

Run:
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverA4.py
   (YPN) jinn@Liu:~/YPN/OPNet$ python train_modelA4.py
Input:
   X_img:   /home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5  (X for debugging)
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

# Set these three global values the same as those in serverA4
DATA_DIR_Imgs = '/home/jinn/dataAll/comma10k/Ximgs_yuv/'  # Ximgs with 10 images only for debugging
DATA_DIR_Msks = '/home/jinn/dataAll/comma10k/Xmasks/'
BATCH_SIZE = 2

EPOCHS = 5

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def gen(hwm, host, port, model):
    for tup in client_generator(hwm=hwm, host=host, port=port):
        X_batch, Y_batch = tup
        print('#---  X_batch[0,   :,  64, 128] =', X_batch[0, :, 64,  128])
        print('#---  Y_batch[0, 128, 256,   :] =', Y_batch[0, 128, 256, :])

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
    print('#---  y_true[0, 128, 256, :] =', y_true[0, 128, 256, :])
    print('#---  y_pred[0, 128, 256, :] =', y_pred[0, 128, 256, :])
      #---  y_true[0,  64, 128, :] = Tensor("custom_loss/strided_slice:0", shape=(None,), dtype=float32)???
      #---  y_true[0,  64, 128, :] = Tensor("custom_loss/strided_slice_1:0", shape=(None,), dtype=float32)???
      # ??? problems are solved by loss = custom_loss(Y_batch, p)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss12 = cce(y_true, y_pred)   # y_true.shape = (2, 256, 512, 12)
      # NumPy Array Slicing: https://www.w3schools.com/python/numpy/trypython.asp?filename=demo_numpy_array_slicing_2d
    loss1 = cce(y_true[:, :, :,  0:6], y_pred[:, :, :,  0:6])
    loss2 = cce(y_true[:, :, :, 6:12], y_pred[:, :, :, 6:12])
    loss = (loss1 + loss2)/2
    print('#---  loss12 =', loss12)
    print('#---  loss1 =', loss1)
    print('#---  loss2 =', loss2)
    print('#---  loss =', loss)
      #---  loss = Tensor("custom_loss/categorical_crossentropy/weighted_loss/value:0", shape=(), dtype=float32)
      #---  loss = tf.Tensor(4.938825, shape=(), dtype=float32)

    return loss

if __name__=="__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Training modelA4')
    parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
    args = parser.parse_args()

    all_img_dirs = os.listdir(DATA_DIR_Imgs)
    all_msk_dirs = os.listdir(DATA_DIR_Msks)
    all_images = [DATA_DIR_Imgs+i for i in all_img_dirs]
    train_len  = int(0.8*len(all_images))
    valid_len  = int(0.2*len(all_images))

    # Build model
    img_shape = (12, 128, 256)
    num_classes = 6
    model = get_model(img_shape, num_classes)
    #model.summary()

    filepath = "./saved_model/modelA4-BestWeights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    #loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    adam = tf.keras.optimizers.Adam(lr=0.0001)

    #model.load_weights('./saved_model/modelA4-BestWeights.hdf5', by_name=True)
    model.compile(optimizer=adam, loss=custom_loss, metrics=["accuracy"])

      # https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    history = model.fit(
      gen(20, args.host, port=args.port, model=model),
      steps_per_epoch=train_len//BATCH_SIZE, epochs=EPOCHS,
      validation_steps=valid_len//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
        # steps_per_epoch = Total Training Samples / Training Batch Size
        # validation_steps = total_validation_samples / validation_batch_size

    #print('#---  # of epochs are run =', len(history.history['loss']))
    model.save('./saved_model/modelA4.h5')

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
