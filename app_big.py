import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import pandas as pd

# ==========================================
# 1. LOAD MODEL & SCALER
# ==========================================
try:
    print("Loading model and scaler...")
    model = joblib.load('big_diabetes_model.pkl')
    scaler = joblib.load('big_scaler.pkl')
    print("✅ System Ready")
except FileNotFoundError:
    messagebox.showerror("Error", "Files not found! Make sure you ran 'train_big_model.py' first.")
    exit()

def predict():
    try:
        # Collect all Yes/No (Checkbox) values
        # .get() returns 1 if checked, 0 if unchecked
        inputs = [
            var_bp.get(),          # HighBP
            var_chol.get(),        # HighChol
            var_cholcheck.get(),   # CholCheck
            float(entry_bmi.get()),# BMI (Body Mass Index)
            var_smoker.get(),      # Smoker
            var_stroke.get(),      # Stroke
            var_heart.get(),       # HeartDiseaseorAttack
            var_phys.get(),        # PhysActivity
            var_fruit.get(),       # Fruits
            var_veg.get(),         # Veggies
            var_alcohol.get(),     # HvyAlcoholConsump
            var_health.get(),      # AnyHealthcare
            var_doc.get(),         # NoDocbcCost
            int(scale_gen.get()),  # GenHlth (1-5)
            int(entry_ment.get()), # MentHlth (0-30 days)
            int(entry_phys.get()), # PhysHlth (0-30 days)
            var_walk.get(),        # DiffWalk
            var_sex.get(),         # Sex (0=Female, 1=Male)
            int(entry_age.get()),  # Age (1-13 scale)
            int(scale_edu.get()),  # Education (1-6)
            int(scale_inc.get())   # Income (1-8)
        ]

        # Reshape for the model (1 row, 21 columns)
        final_input = np.array([inputs])
        
        # Scale the data (Must use the same scaler as training!)
        final_input_scaled = scaler.transform(final_input)

        # Predict
        prediction = model.predict(final_input_scaled)
        prob = model.predict_proba(final_input_scaled)[0][1] # Probability of being diabetic

        # Show Result
        if prediction[0] == 1:
            lbl_result.config(text=f"⚠️ HIGH RISK ({prob*100:.1f}%)", fg="red")
        else:
            lbl_result.config(text=f"🟢 LOW RISK ({prob*100:.1f}%)", fg="green")

    except ValueError:
        messagebox.showerror("Input Error", "Please ensure BMI, Mental Days, and Physical Days are numbers.")

# ==========================================
# GUI LAYOUT
# ==========================================
root = tk.Tk()
root.title("CDC Diabetes Predictor (21 Factors)")
root.geometry("700x650")

# Header
tk.Label(root, text="CDC Health Risk Assessment", font=("Arial", 16, "bold"), fg="teal").pack(pady=10)

# Main container for columns
main_frame = tk.Frame(root)
main_frame.pack(padx=20, pady=10)

# Left Column (Checkboxes)
left_frame = tk.LabelFrame(main_frame, text="Yes / No Questions")
left_frame.grid(row=0, column=0, padx=10, sticky="n")

# Right Column (Values)
right_frame = tk.LabelFrame(main_frame, text="Health Metrics")
right_frame.grid(row=0, column=1, padx=10, sticky="n")

# --- LEFT COLUMN (Checkboxes) ---
def add_check(parent, text):
    var = tk.IntVar()
    chk = tk.Checkbutton(parent, text=text, variable=var, font=("Arial", 10))
    chk.pack(anchor='w', pady=2)
    return var

var_bp        = add_check(left_frame, "High Blood Pressure?")
var_chol      = add_check(left_frame, "High Cholesterol?")
var_cholcheck = add_check(left_frame, "Cholesterol Checked recently?")
var_smoker    = add_check(left_frame, "Have you smoked 100+ cigs?")
var_stroke    = add_check(left_frame, "History of Stroke?")
var_heart     = add_check(left_frame, "Heart Disease or Attack?")
var_phys      = add_check(left_frame, "Physical Activity (past 30 days)?")
var_fruit     = add_check(left_frame, "Eat Fruit daily?")
var_veg       = add_check(left_frame, "Eat Veggies daily?")
var_alcohol   = add_check(left_frame, "Heavy Alcohol Consumption?")
var_health    = add_check(left_frame, "Have Health Insurance?")
var_doc       = add_check(left_frame, "Avoided Doc because of cost?")
var_walk      = add_check(left_frame, "Serious difficulty walking?")
var_sex       = add_check(left_frame, "Are you Male? (Unchecked=Female)")

# --- RIGHT COLUMN (Entries & Sliders) ---
def add_entry(parent, text, default):
    tk.Label(parent, text=text).pack(anchor="w", pady=(5,0))
    entry = tk.Entry(parent)
    entry.insert(0, default)
    entry.pack(fill="x", pady=2)
    return entry

def add_scale(parent, text, min_v, max_v):
    tk.Label(parent, text=text).pack(anchor="w", pady=(5,0))
    scale = tk.Scale(parent, from_=min_v, to=max_v, orient="horizontal")
    scale.set((max_v+min_v)//2) # Set to middle
    scale.pack(fill="x")
    return scale

entry_bmi = add_entry(right_frame, "BMI (Example: 25.0)", "25.0")
scale_gen = add_scale(right_frame, "General Health (1=Excellent, 5=Poor)", 1, 5)
entry_ment = add_entry(right_frame, "Days of Poor Mental Health (0-30)", "0")
entry_phys = add_entry(right_frame, "Days of Physical Illness (0-30)", "0")
entry_age = add_entry(right_frame, "Age Category (1=18yo ... 13=80yo+)", "8")
scale_edu = add_scale(right_frame, "Education Level (1-6)", 1, 6)
scale_inc = add_scale(right_frame, "Income Level (1-8)", 1, 8)

# --- BOTTOM SECTION ---
btn = tk.Button(root, text="ANALYZE RISK", bg="firebrick", fg="white", font=("Arial", 12, "bold"), command=predict)
btn.pack(pady=20, fill="x", padx=50)

lbl_result = tk.Label(root, text="Result: Waiting...", font=("Arial", 18, "bold"))
lbl_result.pack()

root.mainloop()