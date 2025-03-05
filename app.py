import tensorflow
from flask import Flask, request, render_template
import csv
import math
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import PIL
import google.generativeai as genai
import json

app = Flask(__name__)
app.jinja_env.filters['tojson'] = json.dumps

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#os.environ['GOOGLE_API_KEY'] = "AIzaSyCy........................"
#genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

# # define label meaning
label = ['apple pie',
         'baby back ribs',
         'baklava',
         'beef carpaccio',
         'beef tartare',
         'beet salad',
         'beignets',
         'bibimbap',
         'bread pudding',
         'breakfast burrito',
         'bruschetta',
         'caesar salad',
         'cannoli',
         'caprese salad',
         'carrot cake',
         'ceviche',
         'cheese plate',
         'cheesecake',
         'chicken curry',
         'chicken quesadilla',
         'chicken wings',
         'chocolate cake',
         'chocolate mousse',
         'churros',
         'clam chowder',
         'club sandwich',
         'crab cakes',
         'creme brulee',
         'croque madame',
         'cup cakes',
         'deviled eggs',
         'donuts',
         'dumplings',
         'edamame',
         'eggs benedict',
         'escargots',
         'falafel',
         'filet mignon',
         'fish and_chips',
         'foie gras',
         'french fries',
         'french onion soup',
         'french toast',
         'fried calamari',
         'fried rice',
         'frozen yogurt',
         'garlic bread',
         'gnocchi',
         'greek salad',
         'grilled cheese sandwich',
         'grilled salmon',
         'guacamole',
         'gyoza',
         'hamburger',
         'hot and sour soup',
         'hot dog',
         'huevos rancheros',
         'hummus',
         'ice cream',
         'lasagna',
         'lobster bisque',
         'lobster roll sandwich',
         'macaroni and cheese',
         'macarons',
         'miso soup',
         'mussels',
         'nachos',
         'omelette',
         'onion rings',
         'oysters',
         'pad thai',
         'paella',
         'pancakes',
         'panna cotta',
         'peking duck',
         'pho',
         'pizza',
         'pork chop',
         'poutine',
         'prime rib',
         'pulled pork sandwich',
         'ramen',
         'ravioli',
         'red velvet cake',
         'risotto',
         'samosa',
         'sashimi',
         'scallops',
         'seaweed salad',
         'shrimp and grits',
         'spaghetti bolognese',
         'spaghetti carbonara',
         'spring rolls',
         'steak',
         'strawberry shortcake',
         'sushi',
         'tacos',
         'octopus balls',
         'tiramisu',
         'tuna tartare',
         'waffles']

nu_link = 'https://www.nutritionix.com/food/'

# Loading the best saved model to make predictions.
tensorflow.keras.backend.clear_session()
model_best = load_model('models/food_model.h5', compile=False)
print('model successfully loaded!')

start = [0]
passed = [0]
pack = [[]]
num = [0]

nutrients = [
    {'name': 'protein', 'value': 0.0},
    {'name': 'calcium', 'value': 0.0},
    {'name': 'fat', 'value': 0.0},
    {'name': 'carbohydrates', 'value': 0.0},
    {'name': 'vitamins', 'value': 0.0}
]

with open('nutrition101.csv', 'r') as file:
    reader = csv.reader(file)
    nutrition_table = dict()
    for i, row in enumerate(reader):
        if i == 0:
            name = ''
            continue
        else:
            name = row[1].strip()
        protein = float(row[2])
        calcium = float(row[3])
        fat = float(row[4])
        carbs = float(row[5])
        vitamins = float(row[6])
        calories = (protein * 4) + (fat * 9) + (carbs * 4) #calculate calories.
        nutrition_table[name] = [
            {'name': 'protein', 'value': protein},
            {'name': 'calcium', 'value': calcium},
            {'name': 'fat', 'value': fat},
            {'name': 'carbohydrates', 'value': carbs},
            {'name': 'vitamins', 'value': vitamins},
            {'name': 'calories', 'value': calories} #
        ]


@app.route('/')
def index():
    img = 'static/profile.jpg'
    return render_template('index.html', img=img)


@app.route('/recognize')
def magic():
    return render_template('recognize.html', img=file)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.getlist("img")
    for f in file:
        filename = secure_filename(str(num[0] + 500) + '.jpg')
        num[0] += 1
        name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('save name', name)
        f.save(name)

    pack[0] = []
    return render_template('recognize.html', img=file)


@app.route('/predict')
def predict():
    result = []
    # pack = []
    print('total image', num[0])
    if start[0] < num[0]:  # Check if there are images to process
        for i in range(start[0], num[0]):
            pa = dict()
            pa['image'] = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
            # ... (prediction results) ...
            pred_img = tensorflow.keras.preprocessing.image.load_img(pa['image'], target_size=(224, 224))
            pred_img = tensorflow.keras.preprocessing.image.img_to_array(pred_img)
            pred_img = np.expand_dims(pred_img, axis=0)
            pred_img = pred_img / 255.0
            pred = model_best.predict(pred_img)
            if pred is not None and not (math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and math.isnan(pred[0][2]) and math.isnan(pred[0][3])):
                top = pred.argsort()[0][-3:]
                label.sort()
                _true = label[top[2]]
                x = dict()
                x[_true] = float("{:.2f}".format(pred[0][top[2]] * 100))
                x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
                x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))
                pa['result'] = x
                pa['nutrition'] = nutrition_table[_true]
                pa['food'] = f'{nu_link}{_true}'
                pa['idx'] = i - start[0]
                pa['quantity'] = 100

                try:
                    pa['nutrition'] = nutrition_table[_true]
                except KeyError:
                    print(f"Error: Nutrition data not found for {_true}")
                    pa['nutrition'] = [{'name': 'protein', 'value': 0.0},{'name': 'calcium', 'value': 0.0},{'name': 'fat', 'value': 0.0},{'name': 'carbohydrates', 'value': 0.0},{'name': 'vitamins', 'value': 0.0},{'name': 'calories', 'value': 0.0}]

                pack[0].append(pa)
                passed[0] += 1
            else:
                print(f"Prediction failed for image {i+500}.jpg")
                pa = dict()
                pa['image'] = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
                pa['result'] = {"prediction failed": 100.0}
                pa['nutrition'] = [{'name': 'protein', 'value': 0.0},{'name': 'calcium', 'value': 0.0},{'name': 'fat', 'value': 0.0},{'name': 'carbohydrates', 'value': 0.0},{'name': 'vitamins', 'value': 0.0},{'name': 'calories', 'value': 0.0}]
                pa['food'] = ""
                pa['idx'] = i - start[0]
                pa['quantity'] = 100
                pack[0].append(pa)
                passed[0] += 1

        start[0] = passed[0]
        print('successfully packed')

        # Calculate total calories
        total_calories = 0
        for p in pack[0]:
            total_calories += p['nutrition'][5]['value'] #add calories from nutrition table.

        return render_template('results.html', pack=pack[0], total_calories=total_calories)
    else:
        print("No images to process.")
        pack[0].append({"image": "","result": {"No Images uploaded": 100.0}, "nutrition": [{'name': 'protein', 'value': 0.0},{'name': 'calcium', 'value': 0.0},{'name': 'fat', 'value': 0.0},{'name': 'carbohydrates', 'value': 0.0},{'name': 'vitamins', 'value': 0.0},{'name': 'calories', 'value': 0.0}], "food": "", "idx": 0, "quantity": 0})
        return render_template('results.html', pack=pack[0], total_calories=0)

        




if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='127.0.0.1')
    @click.argument('PORT', default=5000, type=int)
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using
            python server.py
        Show the help text using
            python server.py --help
        """
        HOST, PORT = host, port
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
    run()