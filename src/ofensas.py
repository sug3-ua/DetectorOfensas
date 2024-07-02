import re
import numpy as np
import pandas as pd
import nltk
import pickle
import tensorflow as tf
import datetime
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from flask import Flask

app = Flask(__name__)

training_file_path = "dataset/training_set.tsv"
test_file_path = "dataset/test_set.tsv"

training_data = pd.read_csv(training_file_path, delimiter="\t")
test_data = pd.read_csv(test_file_path, delimiter="\t")

nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("spanish",ignore_stopwords=True)
stop_words = set(stopwords.words('spanish'))

longitud_max_tweet = 7000
label_encoder = LabelEncoder()

final_model = Sequential()
final_tokenizer = Tokenizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    tokens = [stemmer.stem(word) for word in tokenizer.tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

training_data['text_prep'] = training_data['comment'].apply(preprocess_text)
test_data['text_prep'] = test_data['comment'].apply(preprocess_text)

def plot_history(history, title):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, 'b-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
    plt.title('Training and validation accuracy for ' + title)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and validation loss for ' + title)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * tf.keras.backend.pow((tf.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(tf.keras.backend.sum(fl, axis=-1))
    return focal_loss_fixed

# Bag of Words
def BoW():
    # 1. Crear el diccionario
    global longitud_max_tweet
    num_pal = longitud_max_tweet
    tok = Tokenizer()
    tok.fit_on_texts(training_data['text_prep'])
    frecuencias_palabras = sorted(tok.word_counts.items(), key=lambda x: x[1], reverse=True) # Obtener las frecuencias de las palabras y ordenarlas
    top_palabras = frecuencias_palabras[:num_pal] # Seleccionar las 1000 palabras más frecuentes
    palabras_top_dict = {palabra[0]: i+1 for i, palabra in enumerate(top_palabras)}  # Crear un nuevo diccionario con solo las 1000 palabras más frecuentes
    tok.word_index = palabras_top_dict

    vectorizador = CountVectorizer(max_features=num_pal)
  
    # 2. Dividir los datos en entrenamiento
    train = training_data
    X_train = vectorizador.fit_transform(train['text_prep']).toarray()
    y_train = train['label']

    test = test_data
    X_test = vectorizador.transform(test['text_prep']).toarray()
    y_test = test['label']
    
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    num_classes = len(label_encoder.classes_)
    y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test_encoded, num_classes=num_classes)

    # 3. Construccion y entrenamiento del modelo
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Softmax para clasificación multiclase, necesario debido a los posibles resultados: OFP OFG NOE NO
    model.compile(optimizer='adam', loss=focal_loss(), metrics=[CategoricalAccuracy()])
    
    X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(X_train, y_train_one_hot, test_size=0.15)
    
    # Entrenar el modelo
    history = model.fit(X_train, y_train_one_hot, epochs=20, batch_size=128, validation_split=0.1, validation_data=(X_val, y_val_one_hot))
    
    plot_history(history, 'BoW')
    
    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test_one_hot)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    return accuracy, model, tok

# Recurrent Neural Network
def RNN():
    global longitud_max_tweet
    longitud_max_tweet = 7000
    num_pal = longitud_max_tweet

    train = training_data
    num_classes = 4
    test = test_data

    # 1. Crear el diccionario
    tokenizador = Tokenizer()
    tokenizador.fit_on_texts(train['text_prep']) # ajustas el tokenizador a los textos de entrenamiento, lo que permite que el tokenizador cree un diccionario de todas las palabras unicas en el conjunto de entrenamiento.
    frecuencias_palabras = sorted(tokenizador.word_counts.items(), key=lambda x: x[1], reverse=True) ## Obtener las frecuencias de las palabras y ordenarlas
    top_palabras = frecuencias_palabras[:num_pal] # Seleccionar las n palabras mas frecuentes
    palabras_top_dict = {palabra[0]: i+1 for i, palabra in enumerate(top_palabras)}  # Crear un nuevo diccionario con solo las palabras mas frecuentes
    tokenizador.word_index = palabras_top_dict
    num_palabra = len(tokenizador.word_index) + 1
    longitud_max_tweet = num_palabra

    # 2. Convertir los textos en vectores numericos
    X_train = tokenizador.texts_to_sequences(train['text_prep'])
    X_test = tokenizador.texts_to_sequences(test['text_prep'])

    longitudes_secuencias = [len(seq) for seq in X_train]
    longitud_max_tweet = int(np.percentile(longitudes_secuencias, 95)) # Obtenemos la longitud mayor del texto
    X_train = pad_sequences(X_train, maxlen=longitud_max_tweet, padding='post') # Para asegurarte de que todas las secuencias tengan la misma longitud mediante el relleno con ceros en la parte posterior
    X_test = pad_sequences(X_test, maxlen=longitud_max_tweet, padding='post')
    y_train = train['label']
    y_test = test['label']
 
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    num_classes = len(label_encoder.classes_)
    y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test_encoded, num_classes=num_classes)

    # 3. Construccion y entrenamiento del modelo
    model = Sequential()
    model.add(Embedding(num_palabra, output_dim=128, input_length=longitud_max_tweet))
    model.add(Dropout(0.5))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss=focal_loss(), metrics=[CategoricalAccuracy()])
    
    X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(X_train, y_train_one_hot, test_size=0.15)
    
    # Entrenar modelo
    history = model.fit(X_train, y_train_one_hot, epochs=10, batch_size=128, validation_data=(X_val, y_val_one_hot))
    plot_history(history, 'RNN')
    
    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test_one_hot)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    return accuracy, model, tokenizador

# Term Frequency-Inverse Document Frequency
def TF_IDF():
    global longitud_max_tweet
    num_pal = longitud_max_tweet

    train = training_data
    num_classes = 4
    test = test_data

    # 1. Vectorización TF-IDF
    tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.7)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train['text_prep']).toarray()

    feature_names = tfidf_vectorizer.get_feature_names_out()
    word_freq = zip(feature_names, X_train_tfidf.sum(axis=0))

    sorted_word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

    top_palabras = sorted_word_freq[:num_pal]
    palabras_top = [palabra[0] for palabra in top_palabras]

    # 2. Crear un nuevo vectorizador TF-IDF con el número deseado de palabras
    tfidf_vectorizer = TfidfVectorizer(vocabulary=palabras_top)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train['text_prep']).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(test['text_prep']).toarray()
    
    longitud_max_tweet = len(tfidf_vectorizer.vocabulary_)

    y_train = train['label']
    y_test = test['label']
    
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    vocabulario = tfidf_vectorizer.vocabulary_

    tok = Tokenizer()
    tok.word_index = vocabulario
    
    num_classes = len(label_encoder.classes_)
    y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test_encoded, num_classes=num_classes)

        
    # 3. Construcción y entrenamiento del modelo
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train_tfidf.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss=focal_loss(), metrics=[CategoricalAccuracy()])
    
    X_train_tfidf, X_val, y_train_one_hot, y_val_one_hot = train_test_split(X_train_tfidf, y_train_one_hot, test_size=0.15)
    
    # Entrenar modelo
    history = model.fit(X_train_tfidf, y_train_one_hot, epochs=10, batch_size=128, validation_data=(X_val, y_val_one_hot))
    plot_history(history, 'TF_IDF')
    
    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test_tfidf, y_test_one_hot)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    return accuracy, model, tok

def guardar_modelo(model, tokenizer, label_encoder, model_path, tokenizer_path, label_encoder_path, config_path, longitud_max_tweet):  
    model.save(model_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    config = {
        'longitud_max_tweet': longitud_max_tweet
    }
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)


def hacer_predicciones(model, X_nuevos, tokenizador):
    X_nuevos_preprocesado = preprocess_text(X_nuevos)
    
    X_nuevos_secuencia = tokenizador.texts_to_sequences([X_nuevos_preprocesado])
    X_nuevos_padded = pad_sequences(X_nuevos_secuencia, maxlen=longitud_max_tweet, padding='post')
    predicciones_probabilidades = model.predict(X_nuevos_padded)
    predicciones = np.argmax(predicciones_probabilidades, axis=1)[0]
    
    print(X_nuevos_preprocesado)
    print(X_nuevos_secuencia)
    print(predicciones)
    
    clases = label_encoder.classes_
    for clase, probabilidad in zip(clases, predicciones_probabilidades[0]):
        print(f'Clase: {clase}, Probabilidad: {probabilidad * 100:.2f}%')
    
    return predicciones

def main():
    global final_model
    global final_tokenizer
    global label_encoder
    global longitud_max_tweet

    #acc, final_model, final_tokenizer = BoW()
    #acc, final_model, final_tokenizer = TF_IDF()
    acc, final_model, final_tokenizer = RNN()
    
    print("Probar predicciones del modelo:")
    while True:
        texto = input()
        if (texto == 'exit'):
            break
        elif (texto == 'save'):
            guardar_modelo(final_model, final_tokenizer, label_encoder, 'model/model.h5', 'model/tokenizer.pkl', 'model/label_encoder.pkl', 'model/config.pkl', longitud_max_tweet)
        else:
            prediccion = hacer_predicciones(final_model, texto, final_tokenizer)
            print(prediccion)
    
main()