from flask import Flask, request, Response
from flask_swagger_ui import get_swaggerui_blueprint
import jsonpickle
from PIL import Image
import io
from kornia import image_to_tensor
import numpy as np
import torch
from kornia.geometry.transform import resize
import kornia.augmentation as K

SWAGGER_URL = "/docs"
app = Flask(__name__)
app.config["DEBUG"] = True
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, "/static/swagger.json", config={"app_name": "DOGCATPRED"}
)

def process_image(img):
    transform = K.container.AugmentationSequential(
            K.Normalize(torch.zeros(1), torch.tensor([255])),
            data_keys=["input"],
            return_transform=False,
            keepdim=False,
        )
    img = image_to_tensor(img).float()
    img = resize(img, (180, 180), align_corners=True)
    img = transform(img)
    return img



@app.route("/", methods=["GET"])
def home():
    return "<h1>Welcome to our nice API</p>"


@app.route("/predict", methods=["POST"])
def predict():
    decoder = {
        0: "dog",
        1: "cat"
    } 
    m = torch.jit.load('models/model_bigboy.pt')
    img = Image.open(io.BytesIO(request.data))
    img = np.array(img)
    img = process_image(img)
    logits = m(img)
    pred = torch.argmax(logits) 
    
    response = {"message": f"Well isn't that a cute {decoder[pred.item()]}", "probabilities":{"cat":logits[0][1].item(), "dog":logits[0][0].item()}} 
    response_pickle = jsonpickle.encode(response)
    return Response(response=response_pickle, status=200, mimetype="application/json")


app.register_blueprint(swaggerui_blueprint)

app.run()
