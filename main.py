from nmt_utils import *
from keras.optimizers import Adam

from model_service import ModelService

if __name__ == '__main__':

    m = 10000
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

    model_service = ModelService()
    model = model_service.model(model_service.Tx, model_service.Ty, model_service.n_a, model_service.n_s,
                                len(human_vocab), len(machine_vocab))
    model.summary()

    optimizer = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, model_service.Tx, model_service.Ty)

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    print("Xoh.shape:", Xoh.shape)
    print("Yoh.shape:", Yoh.shape)

    s0 = np.zeros((m, model_service.n_s))
    c0 = np.zeros((m, model_service.n_s))
    outputs = list(Yoh.swapaxes(0, 1))
    model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

    model.load_weights('models/model.h5')
    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
                'March 3 2001', 'March 3rd 2001', '1 March 2001']

    for example in EXAMPLES:
        source = string_to_int(example, model_service.Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1).T
        source = np.reshape(source, (1, source.shape[0], source.shape[1]))
        prediction = model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]

        print("source:", example)
        print("output:", ''.join(output), "\n")