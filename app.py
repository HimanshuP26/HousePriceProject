import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("house_data.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Create Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])

    prediction = model.predict([[area, bedrooms, bathrooms]])

    return render_template('index.html',
                           prediction_text=f"Predicted Price: ₹ {round(prediction[0],2)}")

# Run app
if __name__ == "__main__":
    app.run(debug=True)