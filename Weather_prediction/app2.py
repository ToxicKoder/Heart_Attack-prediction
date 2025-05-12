from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load both model and scaler
model, scaler = pickle.load(open('weather.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("weather.html")

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    precipitation = float(request.form['precipitation'])

    input_features = np.array([[year, month, day, precipitation]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]

    # Emoji-based message logic
    if prediction == "rain":
        message = "It rained that day ⛈️"
    elif prediction == "snow":
        message = "Snow ❄️"
    elif prediction == "drizzle":
        message = "Drizzle 💧"
    elif prediction == "sun":
        message = "It is sunny 🌞"
    else:
        message = "Unknown weather type"

    return render_template("weather.html", prediction_text=f"Predicted Weather: {prediction}", message=message)

if __name__ == '__main__':
    app.run(debug=True)
