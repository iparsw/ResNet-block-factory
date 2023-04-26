from tensorflow import keras


def ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Add()([x, copy])
    x = keras.layers.Activation("relu")(x)
    return x


def SRResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Add()([x, copy])
    return x


def EDSRResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Add()([x, copy])
    return x


def RB1ResBlock(x, nf: int = 32):
    return ResBlock(x, nf)


def RB2ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Add()([x, copy])
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    return x


def RB3ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Add()([x, copy])
    return x


def RB4ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Add()([x, copy])
    return x


def RB5ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Add()([x, copy])
    return x


def RB6ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Add()([x, copy])
    x = keras.layers.BatchNormalization(x)
    x = keras.layers.Activation("relu")(x)
    return x


def FRResBlock(x, nf: int = 32,
               batchNormalization: bool = True,
               depth: int = 3):
    """Feature Reuse Residual Block"""
    copy1 = x
    x = keras.layers.Conv2D(nf, 1)(x)
    if batchNormalization:
        x = keras.layers.BatchNormalization()(x)
    copy2 = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    for _ in range(depth - 1):
        if batchNormalization:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    if batchNormalization:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Concatenate()([x, copy2])
    x = keras.layers.Add()([x, copy1])
    x = keras.layers.Activation("relu")(x)
    return x


def FRPAResBlock(x, nf: int = 32,
                 batchNormalization: bool = True,
                 depth: int = 3):
    """Feature Reuse Pre-activation Residual Block"""
    copy1 = x
    if batchNormalization:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(nf, 1)(x)
    copy2 = x
    for _ in range(depth):
        if batchNormalization:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Concatenate()([x, copy2])
    x = keras.layers.Add()([x, copy1])

