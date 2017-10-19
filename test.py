from keras.models import Model
from keras.utils import np_utils
from keras.layers import Dense, Embedding, Input, TimeDistributed, LSTM,  \
    Activation
import numpy as np


def main():
    DATA_SIZE = 99
    BATCH_SIZE = 10
    WORD_COUNT = 4
    WORD_LENGTH = 5
    CHAR_VOCAB_SIZE = 50
    NB_CLASSES = 3

    X = np.random.randint(CHAR_VOCAB_SIZE, size=(DATA_SIZE, WORD_COUNT, WORD_LENGTH))
    Y = np.random.randint(NB_CLASSES, size=DATA_SIZE)
    Y = np_utils.to_categorical(Y, NB_CLASSES)

    input = Input(batch_shape=(BATCH_SIZE, WORD_COUNT, WORD_LENGTH, ), dtype='int32')
    embedded = TimeDistributed(Embedding(CHAR_VOCAB_SIZE, 128, input_length=WORD_COUNT))(input)
    char_lstm = TimeDistributed(LSTM(64))(embedded)
    lstm = LSTM(64)(char_lstm)
    dense = Dense(NB_CLASSES, activation='sigmoid')(lstm)
    output = Activation('softmax')(dense)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X = X[:-(len(X) % BATCH_SIZE)]
    Y = Y[:-(len(Y) % BATCH_SIZE)]
    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=5, verbose=0)
    model.evaluate(X, Y, batch_size=BATCH_SIZE)
    model.predict(X, batch_size=1)


if __name__ == "__main__":
    main()