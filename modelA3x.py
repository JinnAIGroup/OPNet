'''   JLL, 2021.9.9, 9.14, 10.9
modelA3x = UNet
UNet from https://keras.io/examples/vision/oxford_pets_image_segmentation/
OP supercombo from https://drive.google.com/file/d/1L8sWgYKtH77K6Kr3FQMETtAWeQNyyb8R/view

1. Use supercombo I/O
2. Task: Image segmentation
3. Input: 2 YUV images with 6 channels = (1, 12, 128, 256)
   #--- inputs.shape = (None, 12, 128, 256)
   #--- x0.shape = (None, 128, 256, 12)  # permutation layer
4. Output: num_classes = 3
   #--- outputs.shape = (None, 128, 256, 3)
Run:
(YPN) jinn@Liu:~/YPN/OPNet$ python modelA3x.py
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
    for filters in [64, 128, 256]:
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

    for filters in [256, 128, 64, 32]:
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

    # Add a per-pixel classification layer
    x = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    return x

# PN will be used in future
def PN(x):
    x1 = layers.Dense(64, activation='relu')(x)
    x2 = layers.Dense(64, activation='relu')(x)
    x3 = layers.Dense(64, activation='relu')(x)
    out1 = layers.Dense(51)(x1)
    out2 = layers.Dense(50)(x2)
    out3 = layers.Dense(5)(x3)
    outputs = layers.Concatenate(axis=-1)([out1, out2, out3])
    #---  x1.shape, x2.shape, x3.shape = (None, 51) (None, 50) (None, 5)
    #--- in modelgen.py, output1, output3, output2 = (None, 51) (None, 50) (None, 5)
    return x

def get_model(img_shape, num_classes):
    inputs = keras.Input(shape=img_shape)
    print('#--- inputs.shape =', inputs.shape)
    x0 = layers.Permute((2, 3, 1))(inputs)
    print('#--- x0.shape =', x0.shape)
    outputs = UNet(x0, num_classes)
    #x = PN(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    print('#--- outputs.shape =', outputs.shape)
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
separable_conv2d_2 (SeparableCo (None, 32, 64, 128)  8896        activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 64, 128)  512         separable_conv2d_2[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 64, 128)  0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
separable_conv2d_3 (SeparableCo (None, 32, 64, 128)  17664       activation_4[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 64, 128)  512         separable_conv2d_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 16, 32, 128)  0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 16, 32, 128)  8320        add[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 16, 32, 128)  0           max_pooling2d_1[0][0]
                                                                 conv2d_2[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 16, 32, 128)  0           add_1[0][0]
__________________________________________________________________________________________________
separable_conv2d_4 (SeparableCo (None, 16, 32, 256)  34176       activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 16, 32, 256)  1024        separable_conv2d_4[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 16, 32, 256)  0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
separable_conv2d_5 (SeparableCo (None, 16, 32, 256)  68096       activation_6[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 16, 32, 256)  1024        separable_conv2d_5[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 8, 16, 256)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 8, 16, 256)   33024       add_1[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 8, 16, 256)   0           max_pooling2d_2[0][0]
                                                                 conv2d_3[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 8, 16, 256)   0           add_2[0][0]
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 8, 16, 256)   590080      activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 8, 16, 256)   1024        conv2d_transpose[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 8, 16, 256)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 8, 16, 256)   590080      activation_8[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 8, 16, 256)   1024        conv2d_transpose_1[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 16, 32, 256)  0           add_2[0][0]
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 16, 32, 256)  0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 16, 32, 256)  65792       up_sampling2d_1[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 16, 32, 256)  0           up_sampling2d[0][0]
                                                                 conv2d_4[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 16, 32, 256)  0           add_3[0][0]
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 16, 32, 128)  295040      activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 16, 32, 128)  512         conv2d_transpose_2[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 16, 32, 128)  0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 16, 32, 128)  147584      activation_10[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 16, 32, 128)  512         conv2d_transpose_3[0][0]
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 32, 64, 256)  0           add_3[0][0]
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 32, 64, 128)  0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 64, 128)  32896       up_sampling2d_3[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 32, 64, 128)  0           up_sampling2d_2[0][0]
                                                                 conv2d_5[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 32, 64, 128)  0           add_4[0][0]
__________________________________________________________________________________________________
conv2d_transpose_4 (Conv2DTrans (None, 32, 64, 64)   73792       activation_11[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 32, 64, 64)   256         conv2d_transpose_4[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 32, 64, 64)   0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
conv2d_transpose_5 (Conv2DTrans (None, 32, 64, 64)   36928       activation_12[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 32, 64, 64)   256         conv2d_transpose_5[0][0]
__________________________________________________________________________________________________
up_sampling2d_5 (UpSampling2D)  (None, 64, 128, 128) 0           add_4[0][0]
__________________________________________________________________________________________________
up_sampling2d_4 (UpSampling2D)  (None, 64, 128, 64)  0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 64, 128, 64)  8256        up_sampling2d_5[0][0]
__________________________________________________________________________________________________
add_5 (Add)                     (None, 64, 128, 64)  0           up_sampling2d_4[0][0]
                                                                 conv2d_6[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 64, 128, 64)  0           add_5[0][0]
__________________________________________________________________________________________________
conv2d_transpose_6 (Conv2DTrans (None, 64, 128, 32)  18464       activation_13[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 64, 128, 32)  128         conv2d_transpose_6[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 64, 128, 32)  0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
conv2d_transpose_7 (Conv2DTrans (None, 64, 128, 32)  9248        activation_14[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 64, 128, 32)  128         conv2d_transpose_7[0][0]
__________________________________________________________________________________________________
up_sampling2d_7 (UpSampling2D)  (None, 128, 256, 64) 0           add_5[0][0]
__________________________________________________________________________________________________
up_sampling2d_6 (UpSampling2D)  (None, 128, 256, 32) 0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 128, 256, 32) 2080        up_sampling2d_7[0][0]
__________________________________________________________________________________________________
add_6 (Add)                     (None, 128, 256, 32) 0           up_sampling2d_6[0][0]
                                                                 conv2d_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 128, 256, 3)  867         add_6[0][0]
==================================================================================================
Total params: 2,061,571
Trainable params: 2,057,795
Non-trainable params: 3,776
'''
