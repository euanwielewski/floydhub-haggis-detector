import os
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Path to Keras model
model_file = 'model.h5'
basepath = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(basepath, 'models', model_file)

# Load your trained model and create prediction function
model = load_model(model_path)
model._make_predict_function()

# Render webapp
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Upload image and predict if haggis or not haggis
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']

        img_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224)) # 224 for ResNet/VGG16, 299 for Xception
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)

        if preds.argmax() == 0:
            x = preds[0,0]*100
            percent = np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
            result = 'Haggis ' + percent + '%'
        elif preds.argmax() == 1:
            x = preds[0,1]*100
            percent = np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
            result = 'Not Haggis ' + percent + '%'

        return result
    return None

if __name__ == '__main__':
    
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()