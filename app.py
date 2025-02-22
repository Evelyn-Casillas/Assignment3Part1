from flask import Flask, render_template, request
import numpy as np
import joblib
import os  # Added this line

app = Flask(__name__)
filename = 'LogisticModel.pkl'

# Load the model (adjust the model to accept 11 features)
model = joblib.load(filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect the 11 features from the form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])
    feature5 = float(request.form['feature5'])
    feature6 = float(request.form['feature6'])
    feature7 = float(request.form['feature7'])
    feature8 = float(request.form['feature8'])
    feature9 = float(request.form['feature9'])
    feature10 = float(request.form['feature10'])
    feature11 = float(request.form['feature11'])

    # Prepare the features for prediction
    features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11]])

    # Predict using the model
    pred = model.predict(features)

    return render_template('index.html', predict=str(pred))

# Modified for Heroku deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001)) #Use 5001 or 5000 for local testing. Heroku will override.
    app.run(host='0.0.0.0', port=port, debug=False) #Debug set to false.