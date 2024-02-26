import json
import os
import re
from dotenv import load_dotenv
from base64 import b64decode
from io import BytesIO

from flask import Flask, send_from_directory
from flask import request, make_response

from predict import make_prediction

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']

MODEL_NAME = 'SimpleCNNModel'


@app.route("/")
def base():
    return send_from_directory('svelte_client/public', 'index.html')


# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('svelte_client/public', path)


@app.route("/api/predict", methods=['POST'])
def get_emotion_prediction():

    response = {}

    data = request.data.decode('utf-8')
    data = json.loads(data)['data']
    img = re.sub('^data:image/.+;base64,', '', data)
    img = BytesIO(b64decode(img))
    emotion, segmentation_bounds = make_prediction(MODEL_NAME, img)

    response['emotion'] = emotion

    if segmentation_bounds is not None:
        response['segmentation_bounds'] = {
            'x': int(segmentation_bounds[0]),
            'y': int(segmentation_bounds[1]),
            'w': int(segmentation_bounds[2]),
            'h': int(segmentation_bounds[3])
        }
    else:
        response['segmentation_bounds'] = None

    response = make_response(response, 200)
    response.mimetype = "application/json"
    return response


if __name__ == '__main__':
    if os.environ['ENVIRONMENT'] == 'dev':
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0', debug=False)
