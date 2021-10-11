'''   JLL, 2021.9.9, 9.14, 10.9
Build modelB3 = UNet + Pose Net (PN) = opUNetPNB3
UNet from https://keras.io/examples/vision/oxford_pets_image_segmentation/
OP PN supercombo from https://drive.google.com/file/d/1L8sWgYKtH77K6Kr3FQMETtAWeQNyyb8R/view

1. Use supercombo I/O
2. Task: Regression for Path Prediction
3. Input: 2 YUV images with 6 channels = (1, 12, 128, 256)
   #--- inputs.shape = (None, 12, 128, 256)
   #--- x0.shape = (None, 128, 256, 12)  # permutation layer
4. Output: mse loss
   #--- outputs.shape = (None, 128, 256, 3)
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

    # Add a per-pixel classification layer (UNet final layer)
    x = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    # Add layers to PN
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
