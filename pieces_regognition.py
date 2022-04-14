import os
import tensorflow as tf
from keras import layers
from keras.callbacks import TensorBoard
import numpy as np


if __name__ == '__main__':
    # odczytanie danych z fodleru
    dataDir = 'images/pieces'
    dataSet = tf.keras.utils.image_dataset_from_directory(dataDir, image_size=(98, 98))

    # podzial danych na zbior uczacy, walidacyjny i tesotwy
    sizeDS = len(dataSet)
    sizeTrain = int(sizeDS * 0.8)
    sizeTest = int(sizeDS * 0.2)
    train = dataSet.take(sizeTrain)
    temp = dataSet.skip(sizeTrain)
    val = temp.skip(sizeTest)
    test = temp.take(sizeTest)

    # lista etykiet na podstawie nazw folderow
    classNames = ['black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_quinn',
                  'black_rook', 'white_bishop', 'white_king', 'white_knight', 'white_pawn',
                  'white_quinn', 'white_rook']

    # przygotawanie danych zmiana wartosci na wartosci od 0 do 1 oraz obrot obrazow
    dataAugmentation = tf.keras.Sequential(
        [
            layers.Rescaling(1./255, input_shape=(98, 98, 3)),
            layers.RandomFlip("horizontal", input_shape=(98, 98, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # stworzneie modelu uczacacego
    model = tf.keras.Sequential([
        dataAugmentation,
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(12)
    ])

    # kompilacja modelu
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # scieszka do plikow i utworzenie tensorboard
    pathToDir = "logs"
    tensorboard = TensorBoard(log_dir=pathToDir, histogram_freq=1, write_graph=True)

    # uczenie i porownanie sieci z zbiorem tesotwym
    epochs = 10
    model.fit(train, validation_data=val, epochs=epochs, shuffle=True, callbacks=[tensorboard])
    _, accuracyTest = model.evaluate(test, verbose=2)
    print(accuracyTest)

    # zapis h5
    dirToSave = os.path.join(os.getcwd(), 'modelSave')
    if not os.path.isdir(dirToSave):
        os.makedirs(dirToSave)
    pathToModelSave = os.path.join(dirToSave, 'model.h5')
    model.save(pathToModelSave)

    test_chess_pieces_dir = 'images/images_to_test_model'
    files = os.listdir(test_chess_pieces_dir)

    for f in files:

        img = tf.keras.utils.load_img(test_chess_pieces_dir+'/'+f, target_size=(98, 98))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(classNames[np.argmax(score)], 100 * np.max(score))
        )

        img.show(title=classNames[np.argmax(score)])