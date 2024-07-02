import re
import numpy as np
import pickle
import nltk
import tensorflow as ts
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from flask import redirect, url_for, Flask, render_template, request
from flask import jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

loaded_model = Sequential()
loaded_tokenizer = Tokenizer()
loaded_label_encoder = LabelEncoder()
loaded_longitud_max_tweet = None

nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("spanish",ignore_stopwords=True)
stop_words = set(stopwords.words('spanish'))

@app.route("/process", methods=["POST"])
def process_text():
    input_text = request.form["input_text"]
    prediccion = hacer_predicciones(loaded_model, input_text, loaded_tokenizer, loaded_longitud_max_tweet)
    if prediccion < 2:
        prediccion = "No ofensivo"
    else:
        prediccion = "Ofensivo"
    return jsonify({'prediccion': prediccion})

@app.route("/")
def index():
    prediccion = request.args.get('prediccion')
    return render_template("index.html", prediccion=prediccion)

def run_flask():
    app.run(debug=True, use_reloader=False)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    tokens = [stemmer.stem(word) for word in tokenizer.tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

def hacer_predicciones(model, X_nuevos, tokenizer, longitud_max_tweet):
    X_nuevos_preprocesado = preprocess_text(X_nuevos)
    
    X_nuevos_secuencia = tokenizer.texts_to_sequences([X_nuevos_preprocesado])
    X_nuevos_padded = pad_sequences(X_nuevos_secuencia, maxlen=longitud_max_tweet, padding='post')
    predicciones_probabilidades = model.predict(X_nuevos_padded)
    predicciones = np.argmax(predicciones_probabilidades, axis=1)[0]
    
    print(X_nuevos_preprocesado)
    print(X_nuevos_secuencia)
    print(predicciones)
    
    clases = loaded_label_encoder.classes_
    
    for clase, probabilidad in zip(clases, predicciones_probabilidades[0]):
        print(f'Clase: {clase}, Probabilidad: {probabilidad * 100:.2f}%')
    
    return predicciones

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

def cargar_modelo(model_path, tokenizer_path, label_encoder_path, config_path):
    model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()}) # Cargar el modelo Keras
    with open(tokenizer_path, 'rb') as f: # Cargar el tokenizador
        tokenizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f: # Cargar el LabelEncoder
        label_encoder = pickle.load(f)
    with open(config_path, 'rb') as f: # Cargar la configuración
        config = pickle.load(f)
    longitud_max_tweet = config['longitud_max_tweet']
    return model, tokenizer, label_encoder, longitud_max_tweet

def main():
    global loaded_model
    global loaded_tokenizer
    global loaded_label_encoder
    global loaded_longitud_max_tweet
    loaded_model, loaded_tokenizer, loaded_label_encoder, loaded_longitud_max_tweet = cargar_modelo('model/model.h5', 'model/tokenizer.pkl', 'model/label_encoder.pkl', 'model/config.pkl')

if __name__ == "__main__":
    main()
    app.run(debug=True)