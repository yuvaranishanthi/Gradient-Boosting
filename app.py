from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature lists
model = joblib.load("model/gb_model.pkl")
numeric_features = joblib.load("model/numeric_features.pkl")
categorical_features = joblib.load("model/categorical_features.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form inputs
            input_data = {}
            for feature in numeric_features:
                input_data[feature] = float(request.form.get(feature))
            for feature in categorical_features:
                input_data[feature] = request.form.get(feature)

            # Create DataFrame
            df_input = pd.DataFrame([input_data])

            # Predict
            predicted_value = model.predict(df_input)[0]
            prediction = f"${predicted_value:,.2f}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html",
                           numeric_features=numeric_features,
                           categorical_features=categorical_features,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
