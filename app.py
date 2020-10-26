# _*_ coding: utf-8 _*_
import io
from PIL import Image

import numpy as np
from flask import Flask
from flask import request
from flask import render_template, redirect, url_for, request
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import random

app = Flask(__name__)


@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('c:/Web/pic/unknow/' + secure_filename(f.filename))
        return '업로드 성공!! 뒤로가기버튼을 누른 후 분석 버튼을 클릭하세요!'


@app.route('/class', methods=['GET', ' POST'])
def predicts():
    inception_url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4'
    feature_model = tf.keras.Sequential([
        hub.KerasLayer(inception_url, output_shape=(2048,), trainable=False)
    ])
    feature_model.build([None, 244, 244, 3])

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        "C:/Web/pic/", target_size=(244, 244), batch_size=32, class_mode='categorical')
    test_features = []

    for idx in range(test_generator.n):
        if idx % 100 == 0:
            print(idx)

        x, _ = test_generator.next()
        feature = feature_model.predict(x)
        test_features.extend(feature)

    test_features = np.array(test_features)

    new_model = keras.models.load_model(
        'plum_classification_v4.h5', compile=False)
    new_model.summary()

    history = new_model.predict(test_features, verbose=1)

    label_text = pd.read_csv('label_v2.csv')
    print(label_text.head())
    label_text['code_name'].nunique()
    unique_Y = label_text['code_name'].unique().tolist()
    train_Y = [unique_Y.index(code_name)
               for code_name in label_text['code_name']]
    train_Y = np.array(train_Y)

    unique_sorted_Y = sorted(unique_Y)
    print(unique_sorted_Y)

    image_path = random.choice(test_generator.filepaths)
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(299, 299))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    feature_vector = feature_model.predict(img)
    prediction = new_model.predict(feature_vector)[0]

    top_5_predict = prediction.argsort()[::-1][:1]
    labels = [unique_sorted_Y[index] for index in top_5_predict]

    return render_template('index.html', answer=labels)


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/lee')
def lee():
    return render_template('lee.html')


if __name__ == "__main__":

    app.run(host="172.31.36.222")
