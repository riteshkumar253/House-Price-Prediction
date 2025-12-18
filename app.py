from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
from datetime import datetime   # ✅ FIXED

app = Flask(__name__)

# Load model & pipeline
model = pickle.load(open("model/house_model.pkl", "rb"))
pipeline = pickle.load(open("model/pipeline.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "longitude": float(request.form["longitude"]),
        "latitude": float(request.form["latitude"]),
        "housing_median_age": float(request.form["housing_median_age"]),
        "total_rooms": float(request.form["total_rooms"]),
        "total_bedrooms": float(request.form["total_bedrooms"]),
        "population": float(request.form["population"]),
        "households": float(request.form["households"]),
        "median_income": float(request.form["median_income"]),
        "ocean_proximity": request.form["ocean_proximity"]
    }

    # ✅ MUST be DataFrame (you already did this correctly)
    df = pd.DataFrame([data])

    # ✅ Apply preprocessing pipeline
    prepared = pipeline.transform(df)

    # ✅ Predict price
    prediction = model.predict(prepared)[0]

    # ✅ Send prediction to HTML
    return render_template(
        "result.html",
        prediction=round(prediction, 2)
    )




@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name = request.form["name"]
        message = request.form["message"]

        with open("feedback.txt", "a") as f:
            f.write(f"\n--- {datetime.now()} ---\n")
            f.write(f"Name: {name}\n")
            f.write(f"Feedback: {message}\n")

        # ✅ Redirect to thank you page
        return redirect(url_for("thank_you"))

    return render_template("feedback.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")



@app.route("/thank_you")
def thank_you():
    return render_template("thank_you.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)







