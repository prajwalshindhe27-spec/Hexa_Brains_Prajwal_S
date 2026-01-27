from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset
data = pd.read_csv("Admission_Predict.csv")

# Fix column spacing issues
data.columns = data.columns.str.strip()

# Features and target
X = data[
    ['GRE Score', 'TOEFL Score', 'University Rating',
     'SOP', 'LOR', 'CGPA', 'Research']
]
y = data['Chance of Admit']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form['gre']),
            float(request.form['toefl']),
            float(request.form['rating']),
            float(request.form['sop']),
            float(request.form['lor']),
            float(request.form['cgpa']),
            float(request.form['research'])
        ]

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0] * 100  # convert to %

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
