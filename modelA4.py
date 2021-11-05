'''   JLL, 2021.11.4
modelA4 = UNet (modified modelA3x)
UNet: https://keras.io/examples/vision/oxford_pets_image_segmentation/
supercombo: https://drive.google.com/file/d/1L8sWgYKtH77K6Kr3FQMETtAWeQNyyb8R/view

1. Use supercombo I/O
2. Task: Multiclass semantic segmentation
3. Input:
   X_batch.shape = (None, 2x6, 128, 256) (num_channels = 6, 2 yuv images)
4. Output:
   Y_pred.shape = (None, 256, 512, 2x6) (num_classes = 6)
Run:
   (YPN) jinn@Liu:~/YPN/OPNet$ python modelA4.py
'''
from tensorflow import keras
from tensorflow.keras import layers

def UNet(x0, num_classes):
    ### [First half of the network: spatial contraction] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x0)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Project residual
        residual = layers.SeparableConv2D(filters, 3, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: spatial expansion] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Project residual
        residual = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    x = layers.Conv2DTranspose(2*num_classes, 3, strides=2, activation="softmax", padding="same")(x)

    return x

def get_model(img_shape, num_classes):
    inputs = keras.Input(shape=img_shape)
    print('#---modelA4 inputs.shape =', inputs.shape)
    x0 = layers.Permute((2, 3, 1))(inputs)
    print('#---modelA4 x0.shape =', x0.shape)
    outputs = UNet(x0, num_classes)

    # Define the model
    model = keras.Model(inputs, outputs)
    print('#---modelA4 outputs.shape =', outputs.shape)
    return model

if __name__=="__main__":
    # Build model
    img_shape = (12, 128, 256)
    num_classes = 6
    model = get_model(img_shape, num_classes)
    model.summary()
'''
'''
