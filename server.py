import json
import re
from base64 import b64decode
from io import BytesIO

from flask import Flask, send_from_directory
from flask import request, make_response

from predict import make_prediction

app = Flask(__name__)

MODEL_PATH = 'model/SimpleCNNModel'


@app.route("/")
def base():
    return send_from_directory('svelte_client/public', 'index.html')


# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('svelte_client/public', path)


@app.route("/api/predict", methods=['POST'])
def get_emotion_prediction():
    data = request.data.decode('utf-8')
    data = json.loads(data)['data']
    img = re.sub('^data:image/.+;base64,', '', data)
    img = BytesIO(b64decode(img))
    emotion = make_prediction(MODEL_PATH, img)
    # print(f"emotion: {emotion}")

    response = make_response(emotion, 200)
    response.mimetype = "text/plain"
    return response


if __name__ == '__main__':
    app.run(debug=True)
