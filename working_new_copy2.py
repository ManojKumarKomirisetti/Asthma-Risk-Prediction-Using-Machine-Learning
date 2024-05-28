import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Function to train models
def train_models(X_train_scaled, y_train):
    # Train Support Vector Machine (SVM) classifier
    svm_clf = SVC(kernel='linear', random_state=42)
    svm_clf.fit(X_train_scaled, y_train)

    # Train Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_scaled, y_train)

    return svm_clf, rf_clf

# Function to predict asthma
def predict_asthma(input_values, scaler, svm_clf, rf_clf):
    input_df = pd.DataFrame([input_values], columns=labels)
    input_scaled = scaler.transform(input_df)
    svm_pred = svm_clf.predict(input_scaled)
    rf_pred = rf_clf.predict(input_scaled)
    return svm_pred, rf_pred

# Create the main window
root = tk.Tk()
root.title("Asthma Detection System")

# Create a frame for the main content
main_frame = ttk.Frame(root, padding="20")
main_frame.grid(row=0, column=0, sticky="nsew")





# Create labels and entry fields for the input features
labels = ['Age (  Above 50 → 5 ,41-50 → 4 ,19-30 → 3 ,31-40 → 2 )', 'Gender  ( Male → 1 , Female → 0)', 'OutdoorJob ( Rarely → 0 ,Occasionally → 1 ,Frequently → 2)', 'OutdoorActivities (Extremely likely → 1 , Neither likely nor unlikely → 2 , Not at all likely → 0)', 'SmokingHabit (Yes → 1 , No → 0)', 
          'Humidity', 'Pressure', 'Temperature', 'UVIndex (Extreme → 1 , Low → 0)', 'WindSpeed']
entries = {}
for i, label in enumerate(labels):
    ttk.Label(main_frame, text=label).grid(row=i, column=0, sticky="w")
    entries[label] = ttk.Entry(main_frame)
    entries[label].grid(row=i, column=1, sticky="ew")

# Load data and train models
data = pd.read_excel("C:/Users/saket/Downloads/modified_dataset (1).xlsx")
X = data[['Age (  Above 50 → 5 ,41-50 → 4 ,19-30 → 3 ,31-40 → 2 )', 'Gender  ( Male → 1 , Female → 0)', 'OutdoorJob ( Rarely → 0 ,Occasionally → 1 ,Frequently → 2)', 'OutdoorActivities (Extremely likely → 1 , Neither likely nor unlikely → 2 , Not at all likely → 0)', 'SmokingHabit (Yes → 1 , No → 0)', 
          'Humidity', 'Pressure', 'Temperature', 'UVIndex (Extreme → 1 , Low → 0)', 'WindSpeed']]
y = data['ACTScore']  # Assuming 'ACTScore' is the target variable
X_train, _, y_train, _, = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled, _, y_train, _, = train_test_split(X, y, test_size=0.2, random_state=42)
svm_clf, rf_clf = train_models(scaler.transform(X_train_scaled), y_train)

# Function to get input values and display results
def display_prediction():
    input_values = [float(entries[label].get()) for label in labels]
    svm_pred, rf_pred = predict_asthma(input_values, scaler, svm_clf, rf_clf)
    svm_label.config(text=f"SVM Prediction: {svm_pred}")
    rf_label.config(text=f"Random Forest Prediction: {rf_pred}")

    avg_pred=(rf_pred+svm_pred)/2
    if 5 <= avg_pred <= 10:
        show_medicines_for_score_5_to_10()
    elif 10 < avg_pred <= 19:
        show_medicines_for_score_10_to_19()
    elif avg_pred > 19:
        show_medicines_for_score_above_19()

# Create a button to trigger prediction
predict_button = ttk.Button(main_frame, text="Predict", command=display_prediction)
predict_button.grid(row=len(labels)+1, columnspan=2, pady=10)

# Create labels to display prediction results
svm_label = ttk.Label(main_frame, text="")
svm_label.grid(row=len(labels)+2, columnspan=2, pady=5)
rf_label = ttk.Label(main_frame, text="")
rf_label.grid(row=len(labels)+3, columnspan=2, pady=5)
def show_medicines_for_score_5_to_10():
    info_text.set("For ACT Score 5 to 10:\n\n"
                  "Medicines:\n"
                  "- Short-Acting Beta Agonists (SABAs)\n"
                  "- Oral Corticosteroids\n"
                  "- Theophylline\n\n"
                  "Natural Remedies:\n"
                  "- Breathing Exercises\n"
                  "- Herbal Remedies\n"
                  "- Honey and Ginger Tea\n\n"
                  "Rare but serious side effects:\n"
                    "-Rapid or irregular heartbeat (palpitations)\n"
                    "-Tremor\n"
                    "-Nervousness or anxiety\n"
                    "-Dizziness\n"
                        "-Muscle cramps\n")

def show_medicines_for_score_10_to_19():
    info_text.set("For ACT Score 10 to 19:\n\n"
                  "Medicines:\n"
                  "- Inhaled Corticosteroids (ICS)\n"
                  "- Combination Inhalers\n"
                  "- Leukotriene Modifiers\n\n"
                  "Natural Remedies:\n"
                  "- Breathing Exercises\n"
                  "- Yoga\n"
                  "- Omega-3 Fatty Acids\n"
                  "- Quercetin\n\n"
                  "Rare but serious side effects:\n"
                "-Adrenal insufficiency (especially with long-term use of high doses)\n"
                "-Osteoporosis (bone thinning)\n"
                "-Growth suppression in children\n"
                "-Glaucoma or cataracts (with prolonged use at high doses)\n"
                "-Increased risk of infections\n")

def show_medicines_for_score_above_19():
    info_text.set("For ACT Score >19:\n\n"
                  "Medicines:\n"
                  "- Long-Acting Beta Agonists (LABAs)\n"
                  "- Biologic Therapies\n"
                  "- Oral Corticosteroids (for exacerbations)\n\n"
                  "Natural Remedies:\n"
                  "- Breathing Exercises\n"
                  "- Yoga\n"
                  "- Acupuncture\n"
                  "- Lifestyle Modifications\n\n"
                  "Rare but serious side effects:\n"
                  "-Sore throat\n"
                    "-Hoarseness or voice changes\n"
                    "-Thrush (oral fungal infection)\n"
                 "-Cough\n"
                    "-Headache\n")
# Information Text
info_text = tk.StringVar()
info_label = ttk.Label(main_frame, textvariable=info_text, wraplength=500)
info_label.grid(row=20, columnspan=2)
# Run the main event loop
root.mainloop()
