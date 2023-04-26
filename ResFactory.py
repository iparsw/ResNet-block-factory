from tensorflow import keras


def ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, copy])
    x = keras.layers.Activation("relu")(x)
    return x


def SRResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
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
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Add()([x, copy])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def RB3ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Add()([x, copy])
    return x


def RB4ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, copy])
    return x


def RB5ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.Add()([x, copy])
    return x


def RB6ResBlock(x, nf: int = 32):
    copy = x
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nf, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, copy])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def FRResBlock(x, nf: int = 32,
               batchNormalization: bool = True,
               depth: int = 3):
    """Feature Reuse Residual Block"""
    copy1 = keras.layers.Concatenate()([x, x])
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
    copy1 = keras.layers.Concatenate()([x, x])
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
    return x


def MLResBlock(x, nf: int = 32,
               number_of_pathes: int = 2,
               depth: int = 1,
               kernal_size: int = 3):
    """Multi Path Residual Block"""
    if nf % number_of_pathes != 0:
        raise ValueError("Number of filters should be divisible by number of pathes")
    copy = x
    pathes = [x] * number_of_pathes
    for i in range(number_of_pathes):
        for _ in range(depth):
            pathes[i] = keras.layers.Conv2D(nf / number_of_pathes, kernel_size=kernal_size, padding="same")(pathes[i])
    x = keras.layers.Concatenate()(pathes)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Add()([x, copy])
    return x
