'''   JLL, 2021.10.19-20
modelA4c = DeepLabV3+ (dilated convolution + ResNet50 + imagenet)
DeepLabV3+ https://keras.io/examples/vision/deeplabv3_plus/
comma10k https://github.com/commaai/comma10k
Input data: comma10k (png)
png imgs = (H, W, C) = (874, 1164, 3); masks = (874, 1164, 1);  imgs2 = (1208, 1928, 3);

1. Task: Multiclass semantic segmentation
   NUM_CLASSES = 5
2. Input:
   Imgs = /home/jinn/dataAll/comma10k/imgs/*.png
   Msks = /home/jinn/dataAll/comma10k/masks/*.png
3. Output:
   plt.title("Training Loss")
   plt.title("Training Accuracy")
   plt.title("Validation Loss")
   plt.title("Validation Accuracy")
   plot_predictions(train_images[:4], colormap, model=model)
     binary mask: one-hot encoded tensor = (?, ?, ?)
     visualize: RGB segmentation masks (each pixel by a unique color corresponding
       to each predicted label from the human_colormap.mat file)
4. Run: (YPN) jinn@Liu:~/YPN/DeepLab$ python modelA4c.py

ValueError: A `Concatenate` layer requires inputs with matching shapes except
for the concat axis. Got inputs shapes: [(None, 165, 219, 256), (None, 219, 291, 48)]
  File "modelA4.py", line 151, in DeeplabV3Plus
    x = layers.Concatenate(axis=-1)([input_a, input_b])
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

    #print('#--- image_tensor.shape =', image_tensor.shape)

# tf.image.resize: images 3-D Tensor of shape [height, width, channels]
#IMAGE_SIZE = 512
IMAGE_H = 874
IMAGE_W = 1164
BATCH_SIZE = 4
NUM_CLASSES = 20
#DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
DATA_DIR_Imgs = "/home/jinn/dataAll/comma10k/imgs"
DATA_DIR_Msks = "/home/jinn/dataAll/comma10k/masks"
NUM_TRAIN_IMAGES = 8  #1000
NUM_VAL_IMAGES = 4  #50
EPOCHS = 2 #25

train_images = sorted(glob(os.path.join(DATA_DIR_Imgs, "*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR_Msks, "*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR_Imgs, "*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR_Msks, "*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
#--- CS1
print('#--- train_images =', train_images)

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_H, IMAGE_W])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_H, IMAGE_W])
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
        kernel_initializer=keras.initializers.HeNormal(),
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

def DeeplabV3Plus(image_h, image_w, num_classes):
    model_input = keras.Input(shape=(image_h, image_w, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_h // 4 // x.shape[1], image_w // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
      # [input_a.shape, input_b.shape] = [(None, 165, 219, 256), (None, 219, 291, 48)]
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_h // x.shape[1], image_w // x.shape[2]),
        interpolation="bilinear",
    )(x)

    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

model = DeeplabV3Plus(image_h=IMAGE_H, image_w=IMAGE_W, num_classes=NUM_CLASSES)
#model.summary()

'''
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

#--- Apollo
https://github.com/ApolloScapeAuto/dataset-api
https://github.com/ApolloScapeAuto/dataset-api/blob/master/lane_segmentation/LanemarkDiscription.pdf
  jpg image = (2710, 3384, 3); label: 35 classes

#--- comma10k
https://github.com/commaai/comma10k
https://blog.comma.ai/crowdsourced-segnet-you-can-help/
(YPN) jinn@Liu:~/YPN$ git clone https://github.com/commaai/comma10k.git

label: 5 labels, in 3 broad classes
  Moves with scene — Road, Lane Markings, Undrivable (including sky)
  Moves themselves — Movable (like vehicles, pedestrians, etc…)
  Moves with you — My car (and anything inside it)

imgs/  -- The PNG image files
masks/ -- PNG segmentation masks (update these!)
imgs2/  -- New PNG image files paired with fisheye PNGs
masks2/ -- PNG segmentation masks (update these!)

Categories of internal segnet
 1 - #402020 - road (all parts, anywhere nobody would look at you funny for driving)
 2 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks)
 3 - #808060 - undrivable
 4 - #00ff66 - movable (vehicles and people/animals)
 5 - #cc00ff - my car (and anything inside it, including wires, mounts, etc. No reflections)

https://github.com/YassineYousfi/comma10k-baseline
  Using U-Net with efficientnet encoder, this baseline reaches 0.044 validation loss.
  This baseline uses two stages (i) 437x582 (ii) 874x1164 (full resolution)
  python3 train_lit_model.py --backbone efficientnet-b4 --version first-stage --gpus 2 --batch-size 28 --epochs 100 --height 437 --width 582
  python3 train_lit_model.py --backbone efficientnet-b4 --version second-stage --gpus 2 --batch-size 7 --learning-rate 5e-5 --epochs 30 --height 874 --width 1164 --augmentation-level hard --seed-from-checkpoint .../efficientnet-b4/first-stage/checkpoints/last.ckpt

https://github.com/qubvel/segmentation_models.pytorch
  High level API (just two lines to create a neural network)
  9 models architectures for binary and multi class segmentation (including legendary Unet)
  113 available encoders
  All encoders have pre-trained weights for faster and better convergence

#--- train_images = [
'/home/jinn/dataAll/comma10k/imgs/0000_0085e9e41513078a_2018-08-19--13-26-08_11_864.png',
'/home/jinn/dataAll/comma10k/imgs/0001_a23b0de0bc12dcba_2018-06-24--00-29-19_17_79.png',
'/home/jinn/dataAll/comma10k/imgs/0002_e8e95b54ed6116a6_2018-09-05--22-04-33_2_608.png',
'/home/jinn/dataAll/comma10k/imgs/0003_97a4ec76e41e8853_2018-09-29--22-46-37_5_585.png',
'/home/jinn/dataAll/comma10k/imgs/0004_2ac95059f70d76eb_2018-05-12--17-46-52_56_371.png',
'/home/jinn/dataAll/comma10k/imgs/0005_836d09212ac1b8fa_2018-06-15--15-57-15_23_345.png',
'/home/jinn/dataAll/comma10k/imgs/0006_0c5c849415c7dba2_2018-08-12--10-26-26_5_1159.png',
'/home/jinn/dataAll/comma10k/imgs/0007_b5e785c1fc446ed0_2018-06-14--08-27-35_78_873.png']
'''
