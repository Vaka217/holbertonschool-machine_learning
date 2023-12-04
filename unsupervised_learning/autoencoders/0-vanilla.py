from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def autoencoder(input_dim, hidden_layers, latent_dim):
    encoder_input = Input(shape=(input_dim,))
    encoder_output = encoder_input
    for units in hidden_layers:
        encoder_output = Dense(units, activation='relu')(encoder_output)

    latent_space = Dense(latent_dim, activation='relu')(encoder_output)
    encoder = Model(encoder_input, latent_space)

    decoder_input = Input(shape=(latent_dim,))
    decoder_output = decoder_input
    for units in reversed(hidden_layers):
        decoder_output = Dense(units, activation='relu')(decoder_output)

    decoder_output = Dense(input_dim, activation='sigmoid')(decoder_output)
    decoder = Model(decoder_input, decoder_output)

    auto_outputs = encoder(encoder_input)
    auto_outputs = decoder(auto_outputs)
    autoencoder = Model(encoder_input, auto_outputs)
    autoencoder.compile(optimizer='adam', loss='mse')

    return encoder, decoder, autoencoder
