import base64
import io
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from dash import Dash, html, dcc, Input, Output
import plotly.express as px

cnn_model = load_model("models/cnn_model.keras")
vgg_model = load_model("models/vgg16_model.keras")

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

with open("histories/cnn_history.json", "r") as file:
    cnn_history = json.load(file)

with open("histories/vgg16_history.json", "r") as file:
    vgg_history = json.load(file)

app = Dash(__name__)

def parse_image(contents):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return image

def prepare_for_cnn(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image).astype("float32")
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def prepare_for_vgg(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image).astype("float32")
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    image_array = np.repeat(image_array, 3, axis=-1)
    image_array = tf.image.resize(image_array, (32, 32))
    return image_array

app.layout = html.Div([
    html.H1("Класифікація зображень Fashion-MNIST", style={"textAlign": "center"}),
    html.H3("Виберіть модель:"),
    dcc.Dropdown(id="model-selector", options=[{"label": "Згорткова нейронна мережа CNN", "value": "cnn"}, {"label": "VGG16", "value": "vgg"}], value="cnn", clearable=False),
    html.Br(),
    dcc.Upload(id="upload-image", children=html.Button("Вибрати зображення"), multiple=False, accept='.jpg,.jpeg,.png,.gif,.bmp'),
    html.Br(),
    html.Div(id="image-container"),
    html.Br(),
    html.H2("Результат класифікації"),
    html.Div(id="prediction-result"),
    dcc.Graph(id="probability-graph"),
    html.H2("Функція втрат під час навчання"),
    dcc.Graph(id="loss-graph"),
    html.H2("Точність під час навчання"),
    dcc.Graph(id="accuracy-graph")
], style={"maxWidth": "1000px", "margin": "auto", "padding": "30px"})

@app.callback(
    Output("image-container", "children"),
    Output("prediction-result", "children"),
    Output("probability-graph", "figure"),
    Input("upload-image", "contents"),
    Input("model-selector", "value")
)
def classify_image(contents, selected_model):
    if contents is None:
        return "", html.P("Спочатку завантажте зображення."), {}

    image = parse_image(contents)

    if selected_model == "cnn":
        input_image = prepare_for_cnn(image)
        predictions = cnn_model.predict(input_image, verbose=0)[0]
        model_name = "CNN"
    else:
        input_image = prepare_for_vgg(image)
        predictions = vgg_model.predict(input_image, verbose=0)[0]
        model_name = "VGG16"

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    probabilities = pd.DataFrame({"Клас": class_names, "Ймовірність (%)": predictions * 100})
    probability_figure = px.bar(probabilities, x="Клас", y="Ймовірність (%)", title=f"Ймовірності класів — {model_name}")

    image_display = html.Img(src=contents, style={"width": "300px", "height": "300px", "objectFit": "contain", "display": "block", "margin": "20px auto"})
    result = html.Div([html.H3(f"Передбачений клас: {predicted_class}"), html.P(f"Ймовірність: {confidence:.2f}%"), html.P(f"Використана модель: {model_name}")])

    return image_display, result, probability_figure

@app.callback(
    Output("loss-graph", "figure"),
    Output("accuracy-graph", "figure"),
    Input("model-selector", "value")
)
def update_training_graphs(selected_model):
    if selected_model == "cnn":
        history_data = cnn_history
        model_name = "CNN"
    else:
        history_data = vgg_history
        model_name = "VGG16"

    epochs = range(1, len(history_data["loss"]) + 1)

    loss_df = pd.DataFrame({"Епоха": list(epochs), "Train": history_data["loss"], "Validation": history_data["val_loss"]})
    loss_figure = px.line(loss_df, x="Епоха", y=["Train", "Validation"], markers=True, title=f"Функція втрат — {model_name}")

    accuracy_df = pd.DataFrame({"Епоха": list(epochs), "Train": history_data["accuracy"], "Validation": history_data["val_accuracy"]})
    accuracy_figure = px.line(accuracy_df, x="Епоха", y=["Train", "Validation"], markers=True, title=f"Точність — {model_name}")

    return loss_figure, accuracy_figure

if __name__ == "__main__":
    app.run(debug=True)