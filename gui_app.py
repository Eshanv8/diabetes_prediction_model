import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd

class DiabetesPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction System")
        self.root.geometry("800x750")
        self.root.configure(bg='#ecf0f1')
        
        try:
            self.model = joblib.load('big_diabetes_model.pkl')
            self.scaler = joblib.load('big_scaler.pkl')
            
            df = pd.read_csv('big_diabetes.csv')
            target_col = 'Diabetes_binary'
            if target_col not in df.columns:
                target_col = df.columns[0]
            self.feature_names = [col for col in df.columns if col != target_col]
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}\n\nPlease run train_big_model.py first!")
            root.destroy()
            return
        
        self.field_labels = {
            'HighBP': 'High Blood Pressure',
            'HighChol': 'High Cholesterol',
            'CholCheck': 'Cholesterol Check (last 5 years)',
            'BMI': 'Body Mass Index (BMI)',
            'Smoker': 'Smoker',
            'Stroke': 'History of Stroke',
            'HeartDiseaseorAttack': 'Heart Disease or Attack',
            'PhysActivity': 'Physical Activity (last 30 days)',
            'Fruits': 'Eat Fruits Daily',
            'Veggies': 'Eat Vegetables Daily',
            'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
            'AnyHealthcare': 'Have Health Insurance',
            'NoDocbcCost': 'Could not see doctor due to cost',
            'GenHlth': 'General Health (1=Excellent, 5=Poor)',
            'MentHlth': 'Mental Health Issues (days in month)',
            'PhysHlth': 'Physical Health Issues (days in month)',
            'DiffWalk': 'Difficulty Walking',
            'Sex': 'Sex (0=Female, 1=Male)',
            'Age': 'Age Category (1-13)',
            'Education': 'Education Level (1-6)',
            'Income': 'Income Level (1-8)'
        }
        
        self.binary_fields = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                             'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                             'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
        
        self.create_widgets()
    
    def create_widgets(self):
        header_frame = tk.Frame(self.root, bg='#3498db', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🏥 Diabetes Risk Assessment", 
                              font=("Arial", 24, "bold"), bg='#3498db', fg='white')
        title_label.pack(pady=20)
        
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        canvas = tk.Canvas(main_frame, bg='#ecf0f1', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#ecf0f1')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.entries = {}
        
        section_frame = tk.LabelFrame(scrollable_frame, text="Patient Information", 
                                     font=("Arial", 12, "bold"), bg='#ecf0f1', 
                                     fg='#2c3e50', padx=15, pady=10)
        section_frame.pack(fill='x', padx=10, pady=10)
        
        for feature in self.feature_names:
            frame = tk.Frame(section_frame, bg='#ecf0f1')
            frame.pack(fill='x', pady=8)
            
            display_name = self.field_labels.get(feature, feature)
            
            label = tk.Label(frame, text=display_name, font=("Arial", 11), 
                           bg='#ecf0f1', fg='#34495e', anchor='w')
            label.pack(side='left', fill='x', expand=True)
            
            if feature in self.binary_fields:
                var = tk.StringVar(value="No")
                yes_rb = tk.Radiobutton(frame, text="Yes", variable=var, value="Yes",
                                       font=("Arial", 10), bg='#ecf0f1', 
                                       selectcolor='#3498db', activebackground='#ecf0f1')
                no_rb = tk.Radiobutton(frame, text="No", variable=var, value="No",
                                      font=("Arial", 10), bg='#ecf0f1',
                                      selectcolor='#3498db', activebackground='#ecf0f1')
                no_rb.pack(side='right', padx=5)
                yes_rb.pack(side='right', padx=5)
                self.entries[feature] = var
            else:
                entry_frame = tk.Frame(frame, bg='white', bd=1, relief='solid')
                entry_frame.pack(side='right')
                entry = tk.Spinbox(entry_frame, from_=0, to=100, font=("Arial", 11), 
                                  width=10, bd=0)
                entry.pack(padx=2, pady=2)
                entry.delete(0, tk.END)
                entry.insert(0, "0")
                self.entries[feature] = entry
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        button_frame = tk.Frame(self.root, bg='#ecf0f1')
        button_frame.pack(pady=15)
        
        predict_btn = tk.Button(button_frame, text="🔍 Analyze Risk", font=("Arial", 13, "bold"),
                               bg='#27ae60', fg='white', padx=40, pady=12,
                               command=self.predict, cursor='hand2', relief='flat',
                               activebackground='#229954')
        predict_btn.pack(side='left', padx=10)
        
        clear_btn = tk.Button(button_frame, text="🔄 Reset All", font=("Arial", 13, "bold"),
                             bg='#7f8c8d', fg='white', padx=40, pady=12,
                             command=self.clear_fields, cursor='hand2', relief='flat',
                             activebackground='#5d6d7e')
        clear_btn.pack(side='left', padx=10)
        
        self.result_frame = tk.Frame(self.root, bg='#ecf0f1')
        self.result_frame.pack(pady=5, padx=20, fill='x')
        
        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 13, "bold"),
                                     bg='#ecf0f1', fg='#2c3e50', wraplength=700, pady=10)
        self.result_label.pack()
    
    def predict(self):
        try:
            input_values = []
            for feature in self.feature_names:
                if feature in self.binary_fields:
                    value = 1 if self.entries[feature].get() == "Yes" else 0
                else:
                    value = float(self.entries[feature].get())
                input_values.append(value)
            
            input_array = np.array(input_values).reshape(1, -1)
            
            input_scaled = self.scaler.transform(input_array)
            
            prediction = self.model.predict(input_scaled)
            probability = self.model.predict_proba(input_scaled)
            
            result_box = tk.Frame(self.result_frame, relief='solid', bd=2)
            result_box.pack(fill='x', pady=5)
            
            for widget in self.result_frame.winfo_children():
                if widget != result_box:
                    widget.destroy()
            
            if prediction[0] == 1:
                risk_prob = probability[0][1] * 100
                result_box.config(bg='#fadbd8')
                icon_label = tk.Label(result_box, text="⚠️", font=("Arial", 40),
                                     bg='#fadbd8', fg='#e74c3c')
                icon_label.pack(pady=10)
                
                risk_label = tk.Label(result_box, text="HIGH RISK OF DIABETES",
                                     font=("Arial", 16, "bold"), bg='#fadbd8', fg='#c0392b')
                risk_label.pack()
                
                prob_label = tk.Label(result_box, text=f"Risk Probability: {risk_prob:.1f}%",
                                     font=("Arial", 13), bg='#fadbd8', fg='#922b21')
                prob_label.pack(pady=5)
                
                advice_label = tk.Label(result_box, 
                                       text="Please consult with a healthcare professional for proper evaluation.",
                                       font=("Arial", 10, "italic"), bg='#fadbd8', fg='#6e2c00',
                                       wraplength=700)
                advice_label.pack(pady=10)
            else:
                healthy_prob = probability[0][0] * 100
                result_box.config(bg='#d5f4e6')
                icon_label = tk.Label(result_box, text="✅", font=("Arial", 40),
                                     bg='#d5f4e6', fg='#27ae60')
                icon_label.pack(pady=10)
                
                risk_label = tk.Label(result_box, text="LOW RISK - Good Health Status",
                                     font=("Arial", 16, "bold"), bg='#d5f4e6', fg='#1e8449')
                risk_label.pack()
                
                prob_label = tk.Label(result_box, text=f"Confidence: {healthy_prob:.1f}%",
                                     font=("Arial", 13), bg='#d5f4e6', fg='#186a3b')
                prob_label.pack(pady=5)
                
                advice_label = tk.Label(result_box, 
                                       text="Maintain a healthy lifestyle with balanced diet and regular exercise.",
                                       font=("Arial", 10, "italic"), bg='#d5f4e6', fg='#0e4f2e',
                                       wraplength=700)
                advice_label.pack(pady=10)
                
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers in all numeric fields!")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def clear_fields(self):
        for feature, widget in self.entries.items():
            if feature in self.binary_fields:
                widget.set("No")
            else:
                widget.delete(0, tk.END)
                widget.insert(0, "0")
        
        for widget in self.result_frame.winfo_children():
            widget.destroy()

def main():
    root = tk.Tk()
    app = DiabetesPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
