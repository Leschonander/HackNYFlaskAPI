from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd


# {"Schooling": 16, "English": 0, "Work Hours"}   
@app.route('/')
def hello_world():
   return 'Hello, World!'

@app.route("/predict", methods=["POST"])
def predict():
   data = pd.read_csv("Cleaned_ASR_Data.csv")

   X = data[['Years_of_Schooling', 'English_Skill_Now']]
   y = data['Income_Per_Hour']

   income_regression = LinearRegression().fit(X, y)

   json = request.json

   result = income_regression.predict([[
      float(json["Schooling"]),
      float(json["English"])
   ]])

   result = {"Result": float(result[0])}

   return jsonify(result)

