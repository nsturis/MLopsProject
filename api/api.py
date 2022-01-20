from flask import Flask, request, Response
from flask_swagger_ui import get_swaggerui_blueprint
import jsonpickle
from PIL import Image
import io
import numpy as np

SWAGGER_URL = "/docs"
app = Flask(__name__)
app.config["DEBUG"] = True
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, "/static/swagger.json", config={"app_name": "DOGCATPRED"}
)


@app.route("/", methods=["GET"])
def home():
    return "<h1>Welcome to our nice API</p>"


@app.route("/predict", methods=["POST"])
def predict():
    img = Image.open(io.BytesIO(request.data))
    img = np.array(img)
    response = {"message": f"Image recieved, shape={img.shape[0]}x{img.shape[1]}"}
    response_pickle = jsonpickle.encode(response)
    return Response(response=response_pickle, status=200, mimetype="application/json")


app.register_blueprint(swaggerui_blueprint)

app.run()
