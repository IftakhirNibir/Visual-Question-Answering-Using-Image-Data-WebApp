from django.shortcuts import render, redirect
from .forms import ImageForm
from .models import ImageClass
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D
from tensorflow import keras
from PIL import Image

#loaded model
loaded_model = keras.models.load_model('vgg16_model.h5')
model = keras.models.load_model('inception_model.h5')

# Load the data from the CSV file
train_data = pd.read_csv('train_data.csv')

# Extract the values from the DataFrame
train_image_ids = train_data['Image_ID'].tolist()
train_questions = train_data['Questions'].tolist()
train_answers = train_data['Answers'].tolist()

# Load the data from the CSV file
test_data = pd.read_csv('test_data.csv')

# Extract the values from the DataFrame
test_image_ids = test_data['Image_ID'].tolist()
test_questions = test_data['Questions'].tolist()
test_answers = test_data['Answers'].tolist()

# Process questions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_questions)
vocab_size = len(tokenizer.word_index) + 1

label_mapping = {'বানর বার': 0, 'বসা': 1, 'না': 2, 'মল': 3, 'ঠিক': 4, 'সাদা': 4, 'হ্যাঁ': 6, 'কুকুর': 7, 'বাদামী': 8, 'দুই': 9, 'চার': 10, 'পাঁচ': 11, 'সবুজ': 12, 'বাজানো': 13, 'বিড়াল': 14, 'নীল': 15, 'টেবিল': 16, 'ছেলে': 17, 'রৌদ্রোজ্জ্বল': 18, 'তিন': 19, 'কালো': 20, 'গাছ': 21, 'এক': 22, 'বেঞ্চ': 23, 'ধূসর': 24, 'হলুদ': 25, 'পাখি': 26, 'মেয়ে': 27, 'মদ': 28, 'মানুষ': 29, 'লাল': 30, 'বই': 31, 'হাড়': 32, 'সকার': 33, 'পাই': 34, 'লগ': 35, 'বেসবল': 36, 'ফুটবল': 37, 'কমলা': 38, 'পালঙ্ক': 39, 'দাঁড়িয়ে': 40, '0': 41, 'মহিলা': 42, 'খাবার': 43, 'মেঝে': 44, 'কাঠবিড়াল': 45, 'পাটি': 46, 'ঘুমাচ্ছেন': 47, 'আপেল': 48, 'বাইক': 49, 'উদ্ভিদ': 50, 'কম্বল': 51, 'ঘাস': 52, 'চেয়ার': 53, 'বাম': 54, 'কিছুই না': 55}

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = img / 255.0
    return img

#Create your views here.

def home(request):
    return render(request,"home.html")


def Vgg(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        txt = request.POST.get('text', '')

        if form.is_valid():
            item = form.save(commit=False)
            item.save()
            obj = form.instance
            input_image_path = "." + obj.image.url
            input_question = txt
            max_question_length = 100
            input_image = preprocess_image(input_image_path)
            input_image = np.expand_dims(input_image, axis=0)
            input_question_sequence = tokenizer.texts_to_sequences([input_question])
            input_question_sequence = pad_sequences(input_question_sequence, maxlen=max_question_length)
            prediction = loaded_model.predict([input_image, input_question_sequence])
            predicted_label = np.argmax(prediction)
            label_mapping_reverse = {v: k for k, v in label_mapping.items()}
            predicted_answer = label_mapping_reverse[predicted_label]
            return render(request, 'vgg.html', {'txt':predicted_answer,'obj': obj, 'form': form})
            
    else:
        form = ImageForm()

    return render(request, 'vgg.html', {'form': form})

def Cnn(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        txt = request.POST.get('text', '')

        if form.is_valid():
            item = form.save(commit=False)
            item.save()
            obj = form.instance
            input_image_path = "." + obj.image.url
            input_question = txt
            max_question_length = 100
            input_image = preprocess_image(input_image_path)
            input_image = np.expand_dims(input_image, axis=0)
            input_question_sequence = tokenizer.texts_to_sequences([input_question])
            input_question_sequence = pad_sequences(input_question_sequence, maxlen=max_question_length)
            prediction = model.predict([input_image, input_question_sequence])
            predicted_label = np.argmax(prediction)
            label_mapping_reverse = {v: k for k, v in label_mapping.items()}
            predicted_answer = label_mapping_reverse[predicted_label]
            return render(request, 'Inception.html', {'txt':predicted_answer,'obj': obj, 'form': form})
            
    else:
        form = ImageForm()

    return render(request, 'Inception.html', {'form': form})

