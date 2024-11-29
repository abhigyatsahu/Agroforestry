from flask import Flask, redirect, render_template
from markupsafe import Markup
import requests
from utils.disease import disease_dic
from utils.model import ResNet9
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import cv2
import  tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

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

@app.route('/')
def disease():
    return render_template('disease.html')

if __name__ == 'main':
    app.run(debug=True)
# -------------------------------------------app-----------------------------------------------------------

@app.route('/disease-predict', methods=['POST'])
def disease_prediction():
    if 'file' not in requests.files:
        return 'No file part'

    file = requests.files['file']

    if file.filename == '':
        return 'No selected file'
    # Save the uploaded file to the 'static' folder
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

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

    cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
    # Test Image Visualization
    import cv2
    image_path = 'test/test/sample1.jpg'

    # Reading an image in default mode
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB

    # Displaying the image
    # plt.imshow(img)
    # plt.title('Test Image')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = cnn.predict(input_arr)

    print(predictions)

    # result_index = np.argmax(predictions) #Return index of max element
    # print(result_index)

    # Displaying the disease prediction
    # model_prediction = class_name[result_index]
    return render_template('disease.html', predictions=predictions)


# ===============================================================================================
if __name__ == 'main':
    app.run(debug=False)