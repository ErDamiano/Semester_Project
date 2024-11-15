import tensorflow as tf
from tensorflow.keras import layers, models

def build_convlstm_model(input_shape):
    """
    Builds a ConvLSTM model for sequence-to-frame prediction.
    """
    model = models.Sequential()

    # ConvLSTM layers for temporal processing
    model.add(layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(layers.BatchNormalization())

    model.add(layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False
    ))
    model.add(layers.BatchNormalization())

    # Convolutional layer to produce final output frame
    model.add(layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation="sigmoid",
        padding="same"
    ))

    # Compile the model
    model.compile(optimizer="adam", loss="mse")
    return model
