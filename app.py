import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        data = pd.read_csv(file_path)
        result = perform_analysis(data)
        return render_template('result.html', result=result)
    return "No file uploaded"

def perform_analysis(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X)
    return prediction

if __name__ == "__main__":
    app.run(debug=True)

