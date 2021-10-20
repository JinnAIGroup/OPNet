'''   JLL, 2021.9.9, 9.14, 10.9, 10.13
Build modelB3a = UNet + Pose Net (PN) = opUNetPNB3
UNet from https://keras.io/examples/vision/oxford_pets_image_segmentation/
OP PN supercombo from https://drive.google.com/file/d/1L8sWgYKtH77K6Kr3FQMETtAWeQNyyb8R/view

1. Use supercombo I/O
2. Task: Regression for Path Prediction
3. Input: 2 YUV images with 6 channels = (1, 12, 128, 256)
   #--- inputs.shape = (None, 12, 128, 256)
   #--- x0.shape = (None, 128, 256, 12)  # permutation layer
4. Output:
   #--- outputs.shape = (None, 112)
Run:
(YPN) jinn@Liu:~/YPN/OPNet$ python modelB3.py
'''
from tensorflow import keras
from tensorflow.keras import layers

def UNet(x0, num_classes):
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x0)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer (UNet final layer)
    x = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Add layers for PN
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)

    return x

# PN will be used in future
def PN(x):
    x1 = layers.Dense(64, activation='relu')(x)
    x2 = layers.Dense(64, activation='relu')(x)
    out1 = layers.Dense(56)(x1)
    out2 = layers.Dense(56)(x2)
    outputs = layers.Concatenate(axis=-1)([out1, out2])
    return outputs

def get_model(img_shape, num_classes):
    inputs = keras.Input(shape=img_shape)
    #--- inputs.shape = (None, 12, 128, 256)
    x0 = layers.Permute((2, 3, 1))(inputs)
    #--- x0.shape = (None, 128, 256, 12)
    x = UNet(x0, num_classes)
    outputs = PN(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    #--- outputs.shape = (None, 112)
    return model

if __name__=="__main__":
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    img_shape = (12, 128, 256)
    num_classes = 3
    model = get_model(img_shape, num_classes)
    model.summary()
'''
10.19 by Tom: the following two lines failed to convert the model to dlc.
up_sampling2d_1 (UpSampling2D)  (None, 64, 128, 64)  0           add[0][0]
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 64, 128, 64)  0           batch_normalization_4[0][0]
---

Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 12, 128, 256 0
__________________________________________________________________________________________________
permute (Permute)               (None, 128, 256, 12) 0           input_1[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 64, 128, 32)  3488        permute[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 64, 128, 32)  128         conv2d[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 64, 128, 32)  0           batch_normalization[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 64, 128, 32)  0           activation[0][0]
__________________________________________________________________________________________________
separable_conv2d (SeparableConv (None, 64, 128, 64)  2400        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 64, 128, 64)  256         separable_conv2d[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 128, 64)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
separable_conv2d_1 (SeparableCo (None, 64, 128, 64)  4736        activation_2[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 64, 128, 64)  256         separable_conv2d_1[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 32, 64, 64)   0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 64, 64)   2112        activation[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 32, 64, 64)   0           max_pooling2d[0][0]
                                                                 conv2d_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 64, 64)   0           add[0][0]
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 32, 64, 64)   36928       activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 64, 64)   256         conv2d_transpose[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 64, 64)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 32, 64, 64)   36928       activation_4[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 64, 64)   256         conv2d_transpose_1[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 64, 128, 64)  0           add[0][0]
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 64, 128, 64)  0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 64, 128, 64)  4160        up_sampling2d_1[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 64, 128, 64)  0           up_sampling2d[0][0]
                                                                 conv2d_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 128, 3)   1731        add_1[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 64, 128, 3)   12          conv2d_3[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 64, 128, 3)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 64, 32)   128         activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 64, 32)   128         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 64, 32)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 16, 32, 64)   2112        activation_6[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 16, 32, 64)   256         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 16, 32, 64)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 8, 16, 128)   8320        activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 8, 16, 128)   512         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 8, 16, 128)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 4, 8, 64)     8256        activation_8[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 4, 8, 64)     256         conv2d_7[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 4, 8, 64)     0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 2, 4, 32)     2080        activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 2, 4, 32)     128         conv2d_8[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 2, 4, 32)     0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 2, 4, 32)     0           activation_10[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 256)          0           activation_11[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 64)           16448       flatten[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           16448       flatten[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 56)           3640        dense[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 56)           3640        dense_1[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 112)          0           dense_2[0][0]
                                                                 dense_3[0][0]
==================================================================================================
Total params: 155,999
Trainable params: 154,777
'''
