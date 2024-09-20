from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necessary for flashing messages

model = pickle.load(open("model.pkl1", "rb"))

# Define the path to your encoders (adjust this path as necessary)
encoder_path = ''


smoking_history_mapping = {
    0: "Never Smoker",
    1: "Former Smoker",
    2: "Current Smoker"
}

# Define columns and corresponding encoders
label_encoders_names = [
    "Gender",
    "Family_History",
    "Comorbidity_Diabetes",
    "Comorbidity_Hypertension",
    "Comorbidity_Heart_Disease",
    "Tumor_Location",
    "Treatment",
    "Comorbidity_Chronic_Lung_Disease",
    "Comorbidity_Kidney_Disease",
    "Comorbidity_Autoimmune_Disease",
    "Comorbidity_Other"
]


# Load Label Encoders
label_encoders = {}
for name in label_encoders_names:
    try:
        with open(f'{encoder_path}label_encoder_{name}.pkl', 'rb') as f:
            label_encoders[name] = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {encoder_path}label_encoder_{name}.pkl")
        # Handle or initialize empty LabelEncoder
        label_encoders[name] = LabelEncoder()


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # try:
        # Extract form data
        age = float(request.form["Age"])
        gender = request.form["Gender"]
        smoking_history = request.form["Smoking_History"]
        tumor_size_mm = float(request.form["Tumor_Size_mm"])
        tumor_location = request.form["Tumor_Location"]
        treatment = request.form["Treatment"]
        family_history = request.form["Family_History"]
        performance_status = float(request.form["Performance_Status"])
        hemoglobin_level = float(request.form["Hemoglobin_Level"])
        white_blood_cell_count = float(request.form["White_Blood_Cell_Count"])
        albumin_level = float(request.form["Albumin_Level"])
        creatinine_level = float(request.form["Creatinine_Level"])
        ldh_level = float(request.form["LDH_Level"])
        calcium_level = float(request.form["Calcium_Level"])
        glucose_level = float(request.form["Glucose_Level"])
        potassium_level = float(request.form["Potassium_Level"])
        sodium_level = float(request.form["Sodium_Level"])

        # Preprocess categorical variables
        try:
            gender_encoded = label_encoders["Gender"].transform([gender])[0]
            tumor_location_encoded = label_encoders["Tumor_Location"].transform([tumor_location])[0]
            treatment_encoded = label_encoders["Treatment"].transform([treatment])[0]
            family_history_encoded = label_encoders["Family_History"].transform([family_history])[0]
        except KeyError as e:
            flash(f"Error: {e} encoder is not properly loaded.")
            return redirect(url_for('home'))



        # Prepare input data for prediction
        input_features = np.array([
            age,
            gender_encoded,
            smoking_history_encoded,
            tumor_size_mm,
            tumor_location_encoded,
            treatment_encoded,
            family_history_encoded,
            performance_status,
            hemoglobin_level,
            white_blood_cell_count,
            albumin_level,
            creatinine_level,
            ldh_level,
            calcium_level,
            glucose_level,
            potassium_level,
            sodium_level
        ]).reshape(1, -1)

        # Predict using the loaded model
        prediction = model.predict(input_features)

        # Render result in the template
        return render_template("result.html", prediction=prediction[0])
    # except Exception as e:
    #     flash(f"An error occurred: {e}")
    #     # return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(port=5000)