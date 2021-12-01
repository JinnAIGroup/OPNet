'''   JLL, 2021.11.28
from /home/jinn/YPN/OPNet/modelB3b.py
Build modelB3 = UNet + Pose Net (PN)
pad Y_batch from 112 to 2383, path vector (pf5P) from 51 to 192 etc.
2383: see https://github.com/JinnAIGroup/OPNet/blob/main/output.txt
outs[0] = pf5P1 + pf5P2 = 385, outs[3] = rf5L1 + rf5L2 = 58
PWYbatch =  2383 - 2*192 - 1 - 2*29 = 1940

1. Use supercombo I/O
2. Task: Regression for Path Prediction
3. Input: 2 YUV images with 6 channels = (none, 12, 128, 256)
   #--- inputs.shape = (None, 12, 128, 256)
   #--- x0.shape = (None, 128, 256, 12)  # permutation layer
4. Output:
   #--- outputs.shape = (None, 2383)
Run:
  (YPN) jinn@Liu:~/YPN/OPNet$ python modelB3.py
'''
from tensorflow import keras
from tensorflow.keras import layers

def UNet(x0, num_classes):
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x0)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)

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

        # Project residual
        residual = layers.Conv2D(filters, 1, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer (UNet final layer)
    x = layers.Conv2D(2*num_classes, 3, activation="softmax", padding="same")(x)

    # Add layers for PN
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)

    return x

# PN will be used in future
def PN(x):
    x1 = layers.Dense(64, activation='relu')(x)
    x2 = layers.Dense(64, activation='relu')(x)
    x3 = layers.Dense(64, activation='relu')(x)
    out1 = layers.Dense(385)(x1)
    out2 = layers.Dense(58)(x2)
    out3 = layers.Dense(1940)(x3)
    outputs = layers.Concatenate(axis=-1)([out1, out2, out3])
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
    #--- outputs.shape = (None, 2383)
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
activation (Activation)         (None, 64, 128, 32)  0           conv2d[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 64, 128, 32)  0           activation[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 64, 128, 64)  18496       activation_1[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 128, 64)  0           conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 64, 128, 64)  36928       activation_2[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 32, 64, 64)   0           conv2d_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 64, 64)   2112        activation[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 32, 64, 64)   0           max_pooling2d[0][0]
                                                                 conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 64, 64)   0           add[0][0]
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 32, 64, 64)   36928       activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 32, 64, 64)   256         conv2d_transpose[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 64, 64)   0           batch_normalization[0][0]
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 32, 64, 64)   36928       activation_4[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 64, 64)   256         conv2d_transpose_1[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 64, 64)   4160        add[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 64, 64)   0           batch_normalization_1[0][0]
                                                                 conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 64, 6)    3462        add_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 64, 6)    24          conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 64, 6)    0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 16, 32, 32)   224         activation_5[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 16, 32, 32)   0           conv2d_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 8, 16, 64)    2112        activation_6[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 8, 16, 64)    0           conv2d_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 4, 8, 128)    8320        activation_7[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 4, 8, 128)    0           conv2d_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 2, 4, 64)     8256        activation_8[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 2, 4, 64)     0           conv2d_9[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 1, 2, 32)     2080        activation_9[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 1, 2, 32)     0           conv2d_10[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 64)           0           activation_10[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 64)           4160        flatten[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           4160        flatten[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 64)           4160        flatten[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 385)          25025       dense[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 58)           3770        dense_1[0][0]
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1940)         126100      dense_2[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 2383)         0           dense_3[0][0]
                                                                 dense_4[0][0]
                                                                 dense_5[0][0]
==================================================================================================
Total params: 331,405
Trainable params: 331,137
Non-trainable params: 268
'''
