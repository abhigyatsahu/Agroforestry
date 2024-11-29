from flask import Flask, redirect, render_template
from markupsafe import Markup
#import requests
from utils.disease import disease_dic
from utils.model import ResNet9
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
import matplotlib.pyplot as plt
import base64


# login and register
# import mysql.connector
from flask_mysqldb import MySQL
from passlib.hash import sha256_crypt
import re
import bcrypt



# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)_Powdery_mildew',
                   'Cherry_(including_sour)_healthy',
                   'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)Common_rust',
                   'Corn_(maize)_Northern_Leaf_Blight',
                   'Corn_(maize)_healthy',
                   'Grape___Black_rot',
                   'Grape__Esca(Black_Measles)',
                   'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange__Haunglongbing(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,bell__Bacterial_spot',
                   'Pepper,bell__healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Custom functions for calculations

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ------------------------------------ FLASK APP -------------------------------------------------


#--app = Flask(name)
#--- from flask import Flask, render_template
#----from datetime import datetime

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'D:\\Major-project\\saved'  # Folder path for saving uploaded files


@app.route('/')
def home():
    return render_template('index.html', title='Flask Template Example')


@app.route('/disease')
def disease():
    return render_template('disease.html')

#if __name__ == '__main__':
 #   app.run(debug=True)

                                 # -------------- login & signup -----------------

app.secret_key = 'your_secret_key'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'major'
mysql = MySQL(app)

# Email validation regex pattern
email_pattern = re.compile(r'^[\w-]+(\.[\w-]+)*@([\w-]+\.)+[a-zA-Z]+$')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        password = request.form['password']


        # Check if email is valid
        if not email_pattern.match(email):
            return 'Invalid email address!'


        # Hashing the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Save user data in the database
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (firstname, lastname, email, password) VALUES (%s, %s, %s, %s)",
                    (firstname, lastname, email, hashed_password))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Retrieve user data from the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user:
            # Check password
            if bcrypt.checkpw(password.encode('utf-8'), user[4].encode('utf-8')):  # Assuming password is the fifth column
                session['email'] = email
                return redirect(url_for('home'))
            else:
                return 'Invalid password!'
        else:
            return 'User not found!'

    return redirect(url_for('static', filename='login.html'))


# -------------------------------------------app-----------------------------------------------------------


@app.route('/disease_predict', methods=['GET', 'POST'])
def disease_predict():

    if 'file' in request.files:
        uploaded_file = request.files['file']
        print('sefsfsfs')
        if uploaded_file.filename != '':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filename)
            print(filename)
           # with open(filename, 'rb') as f:
            #    image_data = f.read()

            validation_set = tf.keras.utils.image_dataset_from_directory(
                'valid',
                labels="inferred",
                label_mode="categorical",
                class_names=None,
                color_mode="rgb",
                batch_size=32,
                image_size=(128, 128),
                shuffle=True,
                seed=None,
                validation_split=None,
                subset=None,
                interpolation="bilinear",
                follow_links=False,
                crop_to_aspect_ratio=False
            )
            class_name = validation_set.class_names


            # Test Image Visualization
            import cv2
            image_path = 'test/test/AppleCedarRust1.JPG'

            # Reading an image in default mode
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
            print('bcxbcxbcxnb')
            #Displaying the image
            #plt.imshow(img)
           # plt.title('Test Image')
            #plt.xticks([])
            #plt.yticks([])
           # plt.show()

            cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
            image = tf.keras.preprocessing.image.load_img(filename, target_size=(128, 128))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            predictions = cnn.predict(input_arr)

            print(predictions)

            result_index = np.argmax(predictions)  # Return index of max element
            print(result_index)

            # Displaying the disease prediction

            model_prediction = class_name[result_index]
            print(model_prediction)

           # plt.imshow(img)
           # plt.title(f"Disease Name: {model_prediction}")
           # plt.xticks([])
           # plt.yticks([])
           # plt.show()



            # Load description from disease dictionary
            description = Markup(str(disease_dic.get(model_prediction, "Description not available")))

            with open(filename, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')


            return render_template('disease.html', model_prediction = model_prediction , description=description, encoded_image=encoded_image, disease_dic=disease_dic)



           # return render_template('disease-result.html', model_prediction = model_prediction)



# ===============================================================================================
if __name__== '__main__':
    app.run(debug=True)