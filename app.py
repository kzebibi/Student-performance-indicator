from flask import Flask, request, render_template  # Import necessary modules

import numpy as np  # Import numpy library
import pandas as pd  # Import pandas library

from sklearn.preprocessing import StandardScaler  # Import StandardScaler from sklearn
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Import CustomData and PredictPipeline classes

app = Flask(__name__)  # Create a Flask application

# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")  # Render the index.html template

@app.route("/predictdata", methods=["GET", "POST"])  # Route for predicting data
def predict_datapoint():
    if request.method == 'GET':  # Check if the request method is GET
        return render_template('home.html')  # Render the home.html template
    else:
        data = CustomData(  # Create a CustomData object with form data
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_data_frame()  # Get data as a DataFrame
        print(pred_df)  # Print the DataFrame
        print("Before Prediction")  # Print message

        predict_pipeline = PredictPipeline()  # Create a PredictPipeline object
        print("Mid Prediction")  # Print message
        results = predict_pipeline.predict(pred_df)  # Make predictions
        print("after Prediction")  # Print message

        return render_template('home.html', results=results[0])  # Render the home.html template with results

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)  # Run the Flask application