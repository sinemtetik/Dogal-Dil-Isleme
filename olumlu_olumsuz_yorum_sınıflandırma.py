# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:33:10 2024

@author: sinem
"""

import pandas as pd
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import warnings

nltk.download('stopwords')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Veri Yükleme ve Hazırlık
df = pd.read_csv(r"...\magaza_yorumlari.csv", encoding="utf-16")
sent_dict = {'Olumsuz': 0, 'Olumlu': 1}
df['Durum'] = df['Durum'].map(sent_dict)

# Metin Temizleme
def clean_text(text):
    unwanted_pattern = r'[!.\n,:“”,?@#"]'
    regex = re.compile(unwanted_pattern)
    cleaned_text = regex.sub(" ", text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    return cleaned_text.lower()  # küçük harfe çevirme

df['new_text'] = df['Görüş'].astype(str).apply(clean_text)

# Gereksiz Kelimelerin Çıkarılması
ineffective = stopwords.words('turkish')
df['new_text'] = df['new_text'].apply(lambda x: " ".join(word for word in x.split() if word not in ineffective))

# Veriyi Eğitim ve Test Seti Olarak Bölme
X_train, X_test, y_train, y_test = train_test_split(df['new_text'], df['Durum'], test_size=0.2, random_state=42)

# Tokenizasyon ve Padleme
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

maxlen = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# LSTM Modelinin Tanımlanması
tf.random.set_seed(42)
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=32),  # Embedding size arttırma
    tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5),  # Overfitting engelleme
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Modelin Derlenmesi
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.summary()

# Modelin Eğitimi
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_lstm = lstm_model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=20, batch_size=32, callbacks=[early_stopping]
)

# Gerçek Zamanlı Tahmin Döngüsü
predict_dict = {0: 'Olumsuz', 1: 'Olumlu'}
while True:
    text = str(input('\nWrite your message: '))
    if text.lower() == 'break':
        break
    cleaned_text = clean_text(text)
    text_seq = tokenizer.texts_to_sequences([cleaned_text])
    text_pad = pad_sequences(text_seq, maxlen=maxlen)
    
    # Tahmin Etme
    pred_dl = (lstm_model.predict(text_pad) > 0.5).astype(int)[0][0]
    print(f'Deep Learning Predict => {predict_dict[pred_dl]}')
