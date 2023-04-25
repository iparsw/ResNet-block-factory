from tensorflow import keras

def ResNetBlock(x, nf=32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Add(x, copy)
    x = keras.layers.Activation("relu")(x)
    return x


def SRResNetBlock(x, nf=32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Add(x, copy)
    return x


def EDSRResNetBlock(x, nf=32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Add(x, copy)
    return x


def RB1ResNetBlock(x, nf=32):
    return ResNetBlock(x, nf)


def RB2ResNetBlock(x, nf=32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Add(x, copy)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    return x


def RB3ResNetBlock(x, nf=32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Add(x, copy)
    return x


def RB4ResNetBlock(x, nf=32):
    copy = x
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Add(x, copy)
    return x


def RB5ResNetBlock(x, nf=32):
    copy = x
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Add(x, copy)
    return x


def RB6ResNetBlock(x, nf=32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Add(x, copy)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    return x