import tensorflow as tf
from keras import layers, models
from keras.layers import Bidirectional,LSTM
def CNN_Model(X_train_tensor_res, y_train_one_hot):
    model = models.Sequential()
    # Convolution
    model.add(layers.Conv1D(128, 3, activation='relu', padding='same', input_shape=(1, 7740)))
    model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
    model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    # Print model summary
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_tensor_res, y_train_one_hot, validation_split=0.1, batch_size=100, epochs=70, verbose=2)
    model.save(f'CNN.keras')
    return model
def RNN_Model(X_train_tensor_res,y_train_one_hot):
    model = tf.keras.Sequential()
    model.add(Bidirectional(LSTM(units=512, return_sequences=True), input_shape=(None, 7740), ))
    model.add(layers.Dropout(0.7))
    model.add(layers.Bidirectional(layers.GRU(512, return_sequences=True), ))
    model.add(layers.Dropout(0.7))
    model.add(layers.Bidirectional(layers.LSTM(units=1024, return_sequences=True)))
    model.add(layers.Dropout(0.9))
    model.add(layers.Bidirectional(layers.LSTM(units=512, return_sequences=True)))
    model.add(Bidirectional(layers.GRU(512)))
    model.add(layers.Dense(7, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_tensor_res, y_train_one_hot, validation_split=0.1, batch_size=100, epochs=70, verbose=2)
    model.save(f'RNN.h5')
    return model