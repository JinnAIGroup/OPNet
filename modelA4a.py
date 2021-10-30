'''   JLL, 2021.10.16-18
modelA4a = DeepLabV3+ (dilated convolution + ResNet50 + imagenet)
Keras DeepLabV3+ https://keras.io/examples/vision/deeplabv3_plus/
Input data: Beijing U (Human Analysis)

1. Task: Multiclass semantic segmentation
   jpg image = (512, 512, 3); mask = (512, 512, 1); NUM_CLASSES = 20
   train_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))
   train_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))
   val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))
2. Input:
   /home/jinn/YPN/DeepLab/instance-level_human_parsing/instance-level_human_parsing/Training
3. Output:
   plt.title("Training Loss")
   plt.title("Training Accuracy")
   plt.title("Validation Loss")
   plt.title("Validation Accuracy")
   plot_predictions(train_images[:4], colormap, model=model)
     binary mask: one-hot encoded tensor = (512, 512, 20)
     visualize: RGB segmentation masks (each pixel by a unique color corresponding
       to each predicted label from the human_colormap.mat file)
4. Run: (YPN) jinn@Liu:~/YPN/DeepLab$ python modelA4a.py
'''
import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 20
DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 10  #1000
NUM_VAL_IMAGES = 5  #50
EPOCHS = 2 #25

train_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
#--- CS1

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image

def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
      #---1 dataset = <TensorSliceDataset shapes: ((), ()), types: (tf.string, tf.string)>
      # from_tensor_slices: Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory.
      # from_tensor_slices((images, labels))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      #---2 dataset = <ParallelMapDataset shapes: ((512, 512, 3), (512, 512, 1)), types: (tf.float32, tf.float32)>
      # num_parallel_calls = the number of batches to compute asynchronously in parallel
      # tf.data.experimental.AUTOTUNE = the number of parallel calls is set dynamically based on available resources.
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
      #---3 dataset = <BatchDataset shapes: ((4, 512, 512, 3), (4, 512, 512, 1)), types: (tf.float32, tf.float32)>
    return dataset

train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),  # https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)

    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()

# Loading the Colormap
colormap = loadmat(
    "./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)
  #--- colormap.shape = (20, 3)
#--- CS3

def infer(model, image_tensor):
      #--- image_tensor.shape = (512, 512, 3)
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
      #---predict predictions.shape = (1, 512, 512, 20)
    predictions = np.squeeze(predictions)
      #---squeeze predictions.shape = (512, 512, 20)
    predictions = np.argmax(predictions, axis=2)
      #---argmax predictions.shape = (512, 512)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
      #--- mask.shape = (512, 512)
    r = np.zeros_like(mask).astype(np.uint8)
      #--- r.shape = (512, 512)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
      #--- rgb.shape = (512, 512, 3)
    return rgb
#--- CS2

def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
          #--- image_tensor.shape = (512, 512, 3)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
          #--- prediction_mask.shape = (512, 512)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
          #--- prediction_colormap = gb.shape = (512, 512, 3)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )

plot_predictions(train_images[:3], colormap, model=model)

'''
    #print('#--- image_tensor.shape =', image_tensor.shape)
#--- CS1
a = ("b", "g", "a", "d", "f", "c", "h", "e")
x = sorted(a)
print(x)
# ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
x = sorted(a)[2:5]
print(x)
# ['c', 'd', 'e']

#--- CS2
import numpy as np
a = np.array([[1, 2, 3],
              [2, 5, 6]])
print('a =', a)
idx = a == 2
print('idx =', idx)
r = np.array([[4, 5, 6],
              [7, 8, 9]])
print("r[idx] =", r[idx])
print("r =", r)

#--- CS3
colormap = loadmat(
    "./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat"
)["colormap"]
#print('#--- colormap1 =', colormap)
colormap = colormap * 100
#print('#--- colormap2 =', colormap)
colormap = colormap.astype(np.uint8)
#print('#--- colormap3 =', colormap)

#--- colormap.shape = (20, 3)
#--- colormap1 = [
 [0.         0.         0.        ]
 [0.5        0.         0.        ]
 [0.99609375 0.         0.        ]
 [0.         0.33203125 0.        ]
 [0.6640625  0.         0.19921875]
 [0.99609375 0.33203125 0.        ] ...

#--- colormap2 = [
 [ 0.        0.        0.      ]
 [50.        0.        0.      ]
 [99.609375  0.        0.      ] ...

#--- colormap3 = [
 [ 0  0  0]* hat (white)
 [50  0  0]* hair (red)
 [99  0  0]* glove (red)
 [ 0 33  0]* sunglasses (green)
 [66  0 19]
 [99 33  0]
 [ 0  0 33]
 [ 0 46 86]
 [33 33  0]* pants
 [ 0 33 33]
 [33 19  0]
 [20 33 50]
 [ 0 50  0]* skirt (green)
 [ 0  0 99]* face (blue)
 [19 66 86]
 [ 0 99 99]
 [33 99 66]
 [66 99 33]
 [99 99  0]
 [99 66  0]]

#--- TUD100
/home/jinn/yolact/eval.py
  np.save('scripts/gt.npy', gt_masks)
/home/jinn/yolact/train.py
  images, targets, masks, num_crowds = prepare_data(datum)

Model: "functional_1"
Architecture: https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 512, 512, 3) 0
__________________________________________________________________________________________________
conv1_pad (ZeroPadding2D)       (None, 518, 518, 3)  0           input_1[0][0]
__________________________________________________________________________________________________
conv1_conv (Conv2D)             (None, 256, 256, 64) 9472        conv1_pad[0][0]
__________________________________________________________________________________________________
conv1_bn (BatchNormalization)   (None, 256, 256, 64) 256         conv1_conv[0][0]
__________________________________________________________________________________________________
conv1_relu (Activation)         (None, 256, 256, 64) 0           conv1_bn[0][0]
__________________________________________________________________________________________________
pool1_pad (ZeroPadding2D)       (None, 258, 258, 64) 0           conv1_relu[0][0]
__________________________________________________________________________________________________
pool1_pool (MaxPooling2D)       (None, 128, 128, 64) 0           pool1_pad[0][0]
__________________________________________________________________________________________________
conv2_block1_1_conv (Conv2D)    (None, 128, 128, 64) 4160        pool1_pool[0][0]
__________________________________________________________________________________________________
conv2_block1_1_bn (BatchNormali (None, 128, 128, 64) 256         conv2_block1_1_conv[0][0]
__________________________________________________________________________________________________
conv2_block1_1_relu (Activation (None, 128, 128, 64) 0           conv2_block1_1_bn[0][0]
__________________________________________________________________________________________________
conv2_block1_2_conv (Conv2D)    (None, 128, 128, 64) 36928       conv2_block1_1_relu[0][0]
__________________________________________________________________________________________________
conv2_block1_2_bn (BatchNormali (None, 128, 128, 64) 256         conv2_block1_2_conv[0][0]
__________________________________________________________________________________________________
conv2_block1_2_relu (Activation (None, 128, 128, 64) 0           conv2_block1_2_bn[0][0]
__________________________________________________________________________________________________
conv2_block1_0_conv (Conv2D)    (None, 128, 128, 256 16640       pool1_pool[0][0]
__________________________________________________________________________________________________
conv2_block1_3_conv (Conv2D)    (None, 128, 128, 256 16640       conv2_block1_2_relu[0][0]
__________________________________________________________________________________________________
conv2_block1_0_bn (BatchNormali (None, 128, 128, 256 1024        conv2_block1_0_conv[0][0]
__________________________________________________________________________________________________
conv2_block1_3_bn (BatchNormali (None, 128, 128, 256 1024        conv2_block1_3_conv[0][0]
__________________________________________________________________________________________________
conv2_block1_add (Add)          (None, 128, 128, 256 0           conv2_block1_0_bn[0][0]
                                                                 conv2_block1_3_bn[0][0]
__________________________________________________________________________________________________
conv2_block1_out (Activation)   (None, 128, 128, 256 0           conv2_block1_add[0][0]
__________________________________________________________________________________________________
conv2_block2_1_conv (Conv2D)    (None, 128, 128, 64) 16448       conv2_block1_out[0][0]
__________________________________________________________________________________________________
conv2_block2_1_bn (BatchNormali (None, 128, 128, 64) 256         conv2_block2_1_conv[0][0]
__________________________________________________________________________________________________
conv2_block2_1_relu (Activation (None, 128, 128, 64) 0           conv2_block2_1_bn[0][0]
__________________________________________________________________________________________________
conv2_block2_2_conv (Conv2D)    (None, 128, 128, 64) 36928       conv2_block2_1_relu[0][0]
__________________________________________________________________________________________________
conv2_block2_2_bn (BatchNormali (None, 128, 128, 64) 256         conv2_block2_2_conv[0][0]
__________________________________________________________________________________________________
conv2_block2_2_relu (Activation (None, 128, 128, 64) 0           conv2_block2_2_bn[0][0]
__________________________________________________________________________________________________
conv2_block2_3_conv (Conv2D)    (None, 128, 128, 256 16640       conv2_block2_2_relu[0][0]
__________________________________________________________________________________________________
conv2_block2_3_bn (BatchNormali (None, 128, 128, 256 1024        conv2_block2_3_conv[0][0]
__________________________________________________________________________________________________
conv2_block2_add (Add)          (None, 128, 128, 256 0           conv2_block1_out[0][0]
                                                                 conv2_block2_3_bn[0][0]
__________________________________________________________________________________________________
conv2_block2_out (Activation)   (None, 128, 128, 256 0           conv2_block2_add[0][0]
__________________________________________________________________________________________________
conv2_block3_1_conv (Conv2D)    (None, 128, 128, 64) 16448       conv2_block2_out[0][0]
__________________________________________________________________________________________________
conv2_block3_1_bn (BatchNormali (None, 128, 128, 64) 256         conv2_block3_1_conv[0][0]
__________________________________________________________________________________________________
conv2_block3_1_relu (Activation (None, 128, 128, 64) 0           conv2_block3_1_bn[0][0]
__________________________________________________________________________________________________
conv2_block3_2_conv (Conv2D)    (None, 128, 128, 64) 36928       conv2_block3_1_relu[0][0]
__________________________________________________________________________________________________
conv2_block3_2_bn (BatchNormali (None, 128, 128, 64) 256         conv2_block3_2_conv[0][0]
__________________________________________________________________________________________________
conv2_block3_2_relu (Activation (None, 128, 128, 64) 0           conv2_block3_2_bn[0][0]
__________________________________________________________________________________________________
conv2_block3_3_conv (Conv2D)    (None, 128, 128, 256 16640       conv2_block3_2_relu[0][0]
__________________________________________________________________________________________________
conv2_block3_3_bn (BatchNormali (None, 128, 128, 256 1024        conv2_block3_3_conv[0][0]
__________________________________________________________________________________________________
conv2_block3_add (Add)          (None, 128, 128, 256 0           conv2_block2_out[0][0]
                                                                 conv2_block3_3_bn[0][0]
__________________________________________________________________________________________________
conv2_block3_out (Activation)   (None, 128, 128, 256 0           conv2_block3_add[0][0]
__________________________________________________________________________________________________
conv3_block1_1_conv (Conv2D)    (None, 64, 64, 128)  32896       conv2_block3_out[0][0]
__________________________________________________________________________________________________
conv3_block1_1_bn (BatchNormali (None, 64, 64, 128)  512         conv3_block1_1_conv[0][0]
__________________________________________________________________________________________________
conv3_block1_1_relu (Activation (None, 64, 64, 128)  0           conv3_block1_1_bn[0][0]
__________________________________________________________________________________________________
conv3_block1_2_conv (Conv2D)    (None, 64, 64, 128)  147584      conv3_block1_1_relu[0][0]
__________________________________________________________________________________________________
conv3_block1_2_bn (BatchNormali (None, 64, 64, 128)  512         conv3_block1_2_conv[0][0]
__________________________________________________________________________________________________
conv3_block1_2_relu (Activation (None, 64, 64, 128)  0           conv3_block1_2_bn[0][0]
__________________________________________________________________________________________________
conv3_block1_0_conv (Conv2D)    (None, 64, 64, 512)  131584      conv2_block3_out[0][0]
__________________________________________________________________________________________________
conv3_block1_3_conv (Conv2D)    (None, 64, 64, 512)  66048       conv3_block1_2_relu[0][0]
__________________________________________________________________________________________________
conv3_block1_0_bn (BatchNormali (None, 64, 64, 512)  2048        conv3_block1_0_conv[0][0]
__________________________________________________________________________________________________
conv3_block1_3_bn (BatchNormali (None, 64, 64, 512)  2048        conv3_block1_3_conv[0][0]
__________________________________________________________________________________________________
conv3_block1_add (Add)          (None, 64, 64, 512)  0           conv3_block1_0_bn[0][0]
                                                                 conv3_block1_3_bn[0][0]
__________________________________________________________________________________________________
conv3_block1_out (Activation)   (None, 64, 64, 512)  0           conv3_block1_add[0][0]
__________________________________________________________________________________________________
conv3_block2_1_conv (Conv2D)    (None, 64, 64, 128)  65664       conv3_block1_out[0][0]
__________________________________________________________________________________________________
conv3_block2_1_bn (BatchNormali (None, 64, 64, 128)  512         conv3_block2_1_conv[0][0]
__________________________________________________________________________________________________
conv3_block2_1_relu (Activation (None, 64, 64, 128)  0           conv3_block2_1_bn[0][0]
__________________________________________________________________________________________________
conv3_block2_2_conv (Conv2D)    (None, 64, 64, 128)  147584      conv3_block2_1_relu[0][0]
__________________________________________________________________________________________________
conv3_block2_2_bn (BatchNormali (None, 64, 64, 128)  512         conv3_block2_2_conv[0][0]
__________________________________________________________________________________________________
conv3_block2_2_relu (Activation (None, 64, 64, 128)  0           conv3_block2_2_bn[0][0]
__________________________________________________________________________________________________
conv3_block2_3_conv (Conv2D)    (None, 64, 64, 512)  66048       conv3_block2_2_relu[0][0]
__________________________________________________________________________________________________
conv3_block2_3_bn (BatchNormali (None, 64, 64, 512)  2048        conv3_block2_3_conv[0][0]
__________________________________________________________________________________________________
conv3_block2_add (Add)          (None, 64, 64, 512)  0           conv3_block1_out[0][0]
                                                                 conv3_block2_3_bn[0][0]
__________________________________________________________________________________________________
conv3_block2_out (Activation)   (None, 64, 64, 512)  0           conv3_block2_add[0][0]
__________________________________________________________________________________________________
conv3_block3_1_conv (Conv2D)    (None, 64, 64, 128)  65664       conv3_block2_out[0][0]
__________________________________________________________________________________________________
conv3_block3_1_bn (BatchNormali (None, 64, 64, 128)  512         conv3_block3_1_conv[0][0]
__________________________________________________________________________________________________
conv3_block3_1_relu (Activation (None, 64, 64, 128)  0           conv3_block3_1_bn[0][0]
__________________________________________________________________________________________________
conv3_block3_2_conv (Conv2D)    (None, 64, 64, 128)  147584      conv3_block3_1_relu[0][0]
__________________________________________________________________________________________________
conv3_block3_2_bn (BatchNormali (None, 64, 64, 128)  512         conv3_block3_2_conv[0][0]
__________________________________________________________________________________________________
conv3_block3_2_relu (Activation (None, 64, 64, 128)  0           conv3_block3_2_bn[0][0]
__________________________________________________________________________________________________
conv3_block3_3_conv (Conv2D)    (None, 64, 64, 512)  66048       conv3_block3_2_relu[0][0]
__________________________________________________________________________________________________
conv3_block3_3_bn (BatchNormali (None, 64, 64, 512)  2048        conv3_block3_3_conv[0][0]
__________________________________________________________________________________________________
conv3_block3_add (Add)          (None, 64, 64, 512)  0           conv3_block2_out[0][0]
                                                                 conv3_block3_3_bn[0][0]
__________________________________________________________________________________________________
conv3_block3_out (Activation)   (None, 64, 64, 512)  0           conv3_block3_add[0][0]
__________________________________________________________________________________________________
conv3_block4_1_conv (Conv2D)    (None, 64, 64, 128)  65664       conv3_block3_out[0][0]
__________________________________________________________________________________________________
conv3_block4_1_bn (BatchNormali (None, 64, 64, 128)  512         conv3_block4_1_conv[0][0]
__________________________________________________________________________________________________
conv3_block4_1_relu (Activation (None, 64, 64, 128)  0           conv3_block4_1_bn[0][0]
__________________________________________________________________________________________________
conv3_block4_2_conv (Conv2D)    (None, 64, 64, 128)  147584      conv3_block4_1_relu[0][0]
__________________________________________________________________________________________________
conv3_block4_2_bn (BatchNormali (None, 64, 64, 128)  512         conv3_block4_2_conv[0][0]
__________________________________________________________________________________________________
conv3_block4_2_relu (Activation (None, 64, 64, 128)  0           conv3_block4_2_bn[0][0]
__________________________________________________________________________________________________
conv3_block4_3_conv (Conv2D)    (None, 64, 64, 512)  66048       conv3_block4_2_relu[0][0]
__________________________________________________________________________________________________
conv3_block4_3_bn (BatchNormali (None, 64, 64, 512)  2048        conv3_block4_3_conv[0][0]
__________________________________________________________________________________________________
conv3_block4_add (Add)          (None, 64, 64, 512)  0           conv3_block3_out[0][0]
                                                                 conv3_block4_3_bn[0][0]
__________________________________________________________________________________________________
conv3_block4_out (Activation)   (None, 64, 64, 512)  0           conv3_block4_add[0][0]
__________________________________________________________________________________________________
conv4_block1_1_conv (Conv2D)    (None, 32, 32, 256)  131328      conv3_block4_out[0][0]
__________________________________________________________________________________________________
conv4_block1_1_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block1_1_conv[0][0]
__________________________________________________________________________________________________
conv4_block1_1_relu (Activation (None, 32, 32, 256)  0           conv4_block1_1_bn[0][0]
__________________________________________________________________________________________________
conv4_block1_2_conv (Conv2D)    (None, 32, 32, 256)  590080      conv4_block1_1_relu[0][0]
__________________________________________________________________________________________________
conv4_block1_2_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block1_2_conv[0][0]
__________________________________________________________________________________________________
conv4_block1_2_relu (Activation (None, 32, 32, 256)  0           conv4_block1_2_bn[0][0]
__________________________________________________________________________________________________
conv4_block1_0_conv (Conv2D)    (None, 32, 32, 1024) 525312      conv3_block4_out[0][0]
__________________________________________________________________________________________________
conv4_block1_3_conv (Conv2D)    (None, 32, 32, 1024) 263168      conv4_block1_2_relu[0][0]
__________________________________________________________________________________________________
conv4_block1_0_bn (BatchNormali (None, 32, 32, 1024) 4096        conv4_block1_0_conv[0][0]
__________________________________________________________________________________________________
conv4_block1_3_bn (BatchNormali (None, 32, 32, 1024) 4096        conv4_block1_3_conv[0][0]
__________________________________________________________________________________________________
conv4_block1_add (Add)          (None, 32, 32, 1024) 0           conv4_block1_0_bn[0][0]
                                                                 conv4_block1_3_bn[0][0]
__________________________________________________________________________________________________
conv4_block1_out (Activation)   (None, 32, 32, 1024) 0           conv4_block1_add[0][0]
__________________________________________________________________________________________________
conv4_block2_1_conv (Conv2D)    (None, 32, 32, 256)  262400      conv4_block1_out[0][0]
__________________________________________________________________________________________________
conv4_block2_1_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block2_1_conv[0][0]
__________________________________________________________________________________________________
conv4_block2_1_relu (Activation (None, 32, 32, 256)  0           conv4_block2_1_bn[0][0]
__________________________________________________________________________________________________
conv4_block2_2_conv (Conv2D)    (None, 32, 32, 256)  590080      conv4_block2_1_relu[0][0]
__________________________________________________________________________________________________
conv4_block2_2_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block2_2_conv[0][0]
__________________________________________________________________________________________________
conv4_block2_2_relu (Activation (None, 32, 32, 256)  0           conv4_block2_2_bn[0][0]
__________________________________________________________________________________________________
conv4_block2_3_conv (Conv2D)    (None, 32, 32, 1024) 263168      conv4_block2_2_relu[0][0]
__________________________________________________________________________________________________
conv4_block2_3_bn (BatchNormali (None, 32, 32, 1024) 4096        conv4_block2_3_conv[0][0]
__________________________________________________________________________________________________
conv4_block2_add (Add)          (None, 32, 32, 1024) 0           conv4_block1_out[0][0]
                                                                 conv4_block2_3_bn[0][0]
__________________________________________________________________________________________________
conv4_block2_out (Activation)   (None, 32, 32, 1024) 0           conv4_block2_add[0][0]
__________________________________________________________________________________________________
conv4_block3_1_conv (Conv2D)    (None, 32, 32, 256)  262400      conv4_block2_out[0][0]
__________________________________________________________________________________________________
conv4_block3_1_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block3_1_conv[0][0]
__________________________________________________________________________________________________
conv4_block3_1_relu (Activation (None, 32, 32, 256)  0           conv4_block3_1_bn[0][0]
__________________________________________________________________________________________________
conv4_block3_2_conv (Conv2D)    (None, 32, 32, 256)  590080      conv4_block3_1_relu[0][0]
__________________________________________________________________________________________________
conv4_block3_2_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block3_2_conv[0][0]
__________________________________________________________________________________________________
conv4_block3_2_relu (Activation (None, 32, 32, 256)  0           conv4_block3_2_bn[0][0]
__________________________________________________________________________________________________
conv4_block3_3_conv (Conv2D)    (None, 32, 32, 1024) 263168      conv4_block3_2_relu[0][0]
__________________________________________________________________________________________________
conv4_block3_3_bn (BatchNormali (None, 32, 32, 1024) 4096        conv4_block3_3_conv[0][0]
__________________________________________________________________________________________________
conv4_block3_add (Add)          (None, 32, 32, 1024) 0           conv4_block2_out[0][0]
                                                                 conv4_block3_3_bn[0][0]
__________________________________________________________________________________________________
conv4_block3_out (Activation)   (None, 32, 32, 1024) 0           conv4_block3_add[0][0]
__________________________________________________________________________________________________
conv4_block4_1_conv (Conv2D)    (None, 32, 32, 256)  262400      conv4_block3_out[0][0]
__________________________________________________________________________________________________
conv4_block4_1_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block4_1_conv[0][0]
__________________________________________________________________________________________________
conv4_block4_1_relu (Activation (None, 32, 32, 256)  0           conv4_block4_1_bn[0][0]
__________________________________________________________________________________________________
conv4_block4_2_conv (Conv2D)    (None, 32, 32, 256)  590080      conv4_block4_1_relu[0][0]
__________________________________________________________________________________________________
conv4_block4_2_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block4_2_conv[0][0]
__________________________________________________________________________________________________
conv4_block4_2_relu (Activation (None, 32, 32, 256)  0           conv4_block4_2_bn[0][0]
__________________________________________________________________________________________________
conv4_block4_3_conv (Conv2D)    (None, 32, 32, 1024) 263168      conv4_block4_2_relu[0][0]
__________________________________________________________________________________________________
conv4_block4_3_bn (BatchNormali (None, 32, 32, 1024) 4096        conv4_block4_3_conv[0][0]
__________________________________________________________________________________________________
conv4_block4_add (Add)          (None, 32, 32, 1024) 0           conv4_block3_out[0][0]
                                                                 conv4_block4_3_bn[0][0]
__________________________________________________________________________________________________
conv4_block4_out (Activation)   (None, 32, 32, 1024) 0           conv4_block4_add[0][0]
__________________________________________________________________________________________________
conv4_block5_1_conv (Conv2D)    (None, 32, 32, 256)  262400      conv4_block4_out[0][0]
__________________________________________________________________________________________________
conv4_block5_1_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block5_1_conv[0][0]
__________________________________________________________________________________________________
conv4_block5_1_relu (Activation (None, 32, 32, 256)  0           conv4_block5_1_bn[0][0]
__________________________________________________________________________________________________
conv4_block5_2_conv (Conv2D)    (None, 32, 32, 256)  590080      conv4_block5_1_relu[0][0]
__________________________________________________________________________________________________
conv4_block5_2_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block5_2_conv[0][0]
__________________________________________________________________________________________________
conv4_block5_2_relu (Activation (None, 32, 32, 256)  0           conv4_block5_2_bn[0][0]
__________________________________________________________________________________________________
conv4_block5_3_conv (Conv2D)    (None, 32, 32, 1024) 263168      conv4_block5_2_relu[0][0]
__________________________________________________________________________________________________
conv4_block5_3_bn (BatchNormali (None, 32, 32, 1024) 4096        conv4_block5_3_conv[0][0]
__________________________________________________________________________________________________
conv4_block5_add (Add)          (None, 32, 32, 1024) 0           conv4_block4_out[0][0]
                                                                 conv4_block5_3_bn[0][0]
__________________________________________________________________________________________________
conv4_block5_out (Activation)   (None, 32, 32, 1024) 0           conv4_block5_add[0][0]
__________________________________________________________________________________________________
conv4_block6_1_conv (Conv2D)    (None, 32, 32, 256)  262400      conv4_block5_out[0][0]
__________________________________________________________________________________________________
conv4_block6_1_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block6_1_conv[0][0]
__________________________________________________________________________________________________
conv4_block6_1_relu (Activation (None, 32, 32, 256)  0           conv4_block6_1_bn[0][0]
__________________________________________________________________________________________________
conv4_block6_2_conv (Conv2D)    (None, 32, 32, 256)  590080      conv4_block6_1_relu[0][0]
__________________________________________________________________________________________________
conv4_block6_2_bn (BatchNormali (None, 32, 32, 256)  1024        conv4_block6_2_conv[0][0]
__________________________________________________________________________________________________
conv4_block6_2_relu (Activation (None, 32, 32, 256)  0           conv4_block6_2_bn[0][0]
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 1, 1, 256)    0           conv4_block6_2_relu[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 1, 1, 256)    65792       average_pooling2d[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 1, 1, 256)    1024        conv2d[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 256)  65536       conv4_block6_2_relu[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 256)  589824      conv4_block6_2_relu[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 256)  589824      conv4_block6_2_relu[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 256)  589824      conv4_block6_2_relu[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu (TensorFlowOpL [(None, 1, 1, 256)]  0           batch_normalization[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 256)  1024        conv2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 256)  1024        conv2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 256)  1024        conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 256)  1024        conv2d_4[0][0]
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 32, 32, 256)  0           tf_op_layer_Relu[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu_1 (TensorFlowO [(None, 32, 32, 256) 0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu_2 (TensorFlowO [(None, 32, 32, 256) 0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu_3 (TensorFlowO [(None, 32, 32, 256) 0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu_4 (TensorFlowO [(None, 32, 32, 256) 0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 32, 32, 1280) 0           up_sampling2d[0][0]
                                                                 tf_op_layer_Relu_1[0][0]
                                                                 tf_op_layer_Relu_2[0][0]
                                                                 tf_op_layer_Relu_3[0][0]
                                                                 tf_op_layer_Relu_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 256)  327680      concatenate[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 256)  1024        conv2d_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 128, 128, 48) 3072        conv2_block3_2_relu[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu_5 (TensorFlowO [(None, 32, 32, 256) 0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 128, 128, 48) 192         conv2d_6[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 128, 128, 256 0           tf_op_layer_Relu_5[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu_6 (TensorFlowO [(None, 128, 128, 48 0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 128, 128, 304 0           up_sampling2d_1[0][0]
                                                                 tf_op_layer_Relu_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 128, 128, 256 700416      concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 128, 128, 256 1024        conv2d_7[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu_7 (TensorFlowO [(None, 128, 128, 25 0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 128, 128, 256 589824      tf_op_layer_Relu_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 128, 128, 256 1024        conv2d_8[0][0]
__________________________________________________________________________________________________
tf_op_layer_Relu_8 (TensorFlowO [(None, 128, 128, 25 0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 512, 512, 256 0           tf_op_layer_Relu_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 512, 512, 20) 5140        up_sampling2d_2[0][0]
==================================================================================================
Total params: 11,857,236
Trainable params: 11,824,500
Non-trainable params: 32,736

'''
