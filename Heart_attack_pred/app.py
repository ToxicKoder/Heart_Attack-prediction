import pickle
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

with open('heart_attack.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

authorized_users = {
    "Aditya Asutosh": "2003-06-15",
    "John Doe": "1990-01-01"
}

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        dob = request.form['dob']
        if name in authorized_users and authorized_users[name] == dob:
            return redirect(url_for('predict'))
        else:
            return render_template('login.html', error="Invalid Name or Date of Birth.")
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            heart_rate = float(request.form['heart_rate'])
            blood_sugar = float(request.form['blood_sugar'])

            input_data = [[age, heart_rate, blood_sugar]]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            print("DEBUG - Prediction output from model:", prediction)

            if prediction == 1 or prediction == 'positive':
                result = "High Risk (Positive)"
            elif prediction == 0 or prediction == 'negative':
                result = "Low Risk (Negative)"
            else:
                result = f"Unexpected prediction value: {prediction}"

        except ValueError:
            result = "Invalid input. Please enter numeric values."

    return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
 