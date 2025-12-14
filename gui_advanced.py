import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sqlite3
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import inch
import os

class AdvancedDiabetesPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Diabetes Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#ecf0f1')
        
        self.init_database()
        
        try:
            if os.path.exists('best_diabetes_model.pkl'):
                self.model = joblib.load('best_diabetes_model.pkl')
                self.scaler = joblib.load('advanced_scaler.pkl')
                self.model_info = joblib.load('model_info.pkl')
                self.all_models = joblib.load('all_models.pkl')
            else:
                self.model = joblib.load('big_diabetes_model.pkl')
                self.scaler = joblib.load('big_scaler.pkl')
                self.model_info = {'best_model': 'Random Forest', 'accuracy': 0.0}
                self.all_models = {'Random Forest': self.model}
            
            df = pd.read_csv('big_diabetes.csv')
            target_col = 'Diabetes_binary'
            if target_col not in df.columns:
                target_col = df.columns[0]
            self.feature_names = [col for col in df.columns if col != target_col]
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
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
            'GenHlth': 'General Health',
            'MentHlth': 'Mental Health Issues (days/month 0-30)',
            'PhysHlth': 'Physical Health Issues (days/month 0-30)',
            'DiffWalk': 'Difficulty Walking',
            'Sex': 'Gender',
            'Age': 'Age Group',
            'Education': 'Education Level',
            'Income': 'Annual Income (Per Year)',
            'Height': 'Height (meters)',
            'Weight': 'Weight (kg)'
        }
        
        self.binary_fields = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                             'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                             'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk']
        
        self.dropdown_fields = {
            'Sex': ['Female', 'Male'],
            'Age': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
                   '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
            'Education': ['Never attended school', 'Elementary', 'Some high school', 
                         'High school graduate', 'Some college', 'College graduate'],
            'Income': ['Less than Rs.1,000,000/year', 'Rs.1,000,000-1,500,000/year', 
                      'Rs.1,500,000-2,000,000/year', 'Rs.2,000,000-2,500,000/year', 
                      'Rs.2,500,000-3,500,000/year', 'Rs.3,500,000-5,000,000/year', 
                      'Rs.5,000,000-7,500,000/year', 'Rs.7,500,000 or more/year'],
            'GenHlth': ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
        }
        
        self.bmi_fields = ['Height', 'Weight']
        
        self.current_patient_name = tk.StringVar()
        self.dark_mode = False
        
        self.colors = {
            'light': {
                'bg': '#ecf0f1',
                'header': '#2c3e50',
                'header_text': 'white',
                'frame_bg': '#ecf0f1',
                'label_fg': '#34495e',
            },
            'dark': {
                'bg': '#1e1e1e',
                'header': '#0d1117',
                'header_text': '#c9d1d9',
                'frame_bg': '#2d2d2d',
                'label_fg': '#c9d1d9',
            }
        }
        
        self.create_menu()
        self.create_widgets()
    
    def init_database(self):
        self.conn = sqlite3.connect('patient_records.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                date TEXT,
                prediction INTEGER,
                probability REAL,
                data TEXT
            )
        ''')
        self.conn.commit()
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Batch Prediction", command=self.batch_prediction)
        file_menu.add_command(label="Export to PDF", command=self.export_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Patient History", command=self.show_history)
        view_menu.add_command(label="Model Comparison", command=self.show_model_comparison)
        view_menu.add_command(label="Feature Importance", command=self.show_feature_importance)
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Dark Mode", command=self.toggle_dark_mode)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_widgets(self):
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=100)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🏥 Advanced Diabetes Risk Assessment System", 
                              font=("Arial", 22, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=10)
        
        model_info_label = tk.Label(header_frame, 
                                    text=f"Model: {self.model_info.get('best_model', 'N/A')} | "
                                         f"Accuracy: {self.model_info.get('accuracy', 0)*100:.2f}%",
                                    font=("Arial", 10), bg='#2c3e50', fg='#ecf0f1')
        model_info_label.pack()
        
        main_container = tk.Frame(self.root, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        left_frame = tk.Frame(main_container, bg='#ecf0f1')
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        name_frame = tk.Frame(left_frame, bg='#ecf0f1')
        name_frame.pack(fill='x', pady=5)
        tk.Label(name_frame, text="Patient Name:", font=("Arial", 11, "bold"),
                bg='#ecf0f1').pack(side='left', padx=5)
        tk.Entry(name_frame, textvariable=self.current_patient_name, 
                font=("Arial", 11), width=30).pack(side='left', padx=5)
        
        canvas_frame = tk.Frame(left_frame, bg='#ecf0f1')
        canvas_frame.pack(fill='both', expand=True, pady=5)
        
        canvas = tk.Canvas(canvas_frame, bg='#ecf0f1', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#ecf0f1')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.entries = {}
        
        section_frame = tk.LabelFrame(scrollable_frame, text="📋 Patient Information", 
                                     font=("Arial", 12, "bold"), bg='#ecf0f1', 
                                     fg='#2c3e50', padx=15, pady=10)
        section_frame.pack(fill='x', padx=10, pady=10)
        
        fields_to_show = ['Height', 'Weight'] + [f for f in self.feature_names if f != 'BMI']
        
        for feature in fields_to_show:
            frame = tk.Frame(section_frame, bg='#ecf0f1')
            frame.pack(fill='x', pady=6)
            
            display_name = self.field_labels.get(feature, feature)
            
            label = tk.Label(frame, text=display_name, font=("Arial", 10), 
                           bg='#ecf0f1', fg='#34495e', anchor='w')
            label.pack(side='left', fill='x', expand=True)
            
            if feature in self.binary_fields:
                var = tk.StringVar(value="No")
                yes_rb = tk.Radiobutton(frame, text="Yes", variable=var, value="Yes",
                                       font=("Arial", 9), bg='#ecf0f1', 
                                       selectcolor='#3498db', activebackground='#ecf0f1')
                no_rb = tk.Radiobutton(frame, text="No", variable=var, value="No",
                                      font=("Arial", 9), bg='#ecf0f1',
                                      selectcolor='#3498db', activebackground='#ecf0f1')
                no_rb.pack(side='right', padx=5)
                yes_rb.pack(side='right', padx=5)
                self.entries[feature] = var
            elif feature in self.dropdown_fields:
                var = tk.StringVar(value=self.dropdown_fields[feature][0])
                dropdown = ttk.Combobox(frame, textvariable=var, 
                                       values=self.dropdown_fields[feature],
                                       font=("Arial", 9), width=18, state='readonly')
                dropdown.pack(side='right', padx=5)
                self.entries[feature] = var
            elif feature in self.bmi_fields:
                entry_frame = tk.Frame(frame, bg='white', bd=1, relief='solid')
                entry_frame.pack(side='right')
                if feature == 'Height':
                    entry = tk.Spinbox(entry_frame, from_=1.0, to=2.5, increment=0.01,
                                      font=("Arial", 10), width=10, bd=0, format="%.2f")
                    entry.delete(0, tk.END)
                    entry.insert(0, "1.70")
                else:
                    entry = tk.Spinbox(entry_frame, from_=30, to=200, increment=1,
                                      font=("Arial", 10), width=10, bd=0)
                    entry.delete(0, tk.END)
                    entry.insert(0, "70")
                entry.pack(padx=2, pady=2)
                self.entries[feature] = entry
            else:
                entry_frame = tk.Frame(frame, bg='white', bd=1, relief='solid')
                entry_frame.pack(side='right')
                entry = tk.Spinbox(entry_frame, from_=0, to=100, font=("Arial", 10), 
                                  width=10, bd=0)
                entry.pack(padx=2, pady=2)
                entry.delete(0, tk.END)
                entry.insert(0, "0")
                self.entries[feature] = entry
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        button_frame = tk.Frame(left_frame, bg='#ecf0f1')
        button_frame.pack(pady=10)
        
        predict_btn = tk.Button(button_frame, text="🔍 Analyze Risk", font=("Arial", 12, "bold"),
                               bg='#27ae60', fg='white', padx=30, pady=10,
                               command=self.predict, cursor='hand2', relief='flat')
        predict_btn.pack(side='left', padx=5)
        
        save_btn = tk.Button(button_frame, text="💾 Save Record", font=("Arial", 12, "bold"),
                            bg='#3498db', fg='white', padx=30, pady=10,
                            command=self.save_record, cursor='hand2', relief='flat')
        save_btn.pack(side='left', padx=5)
        
        clear_btn = tk.Button(button_frame, text="🔄 Clear", font=("Arial", 12, "bold"),
                             bg='#7f8c8d', fg='white', padx=30, pady=10,
                             command=self.clear_fields, cursor='hand2', relief='flat')
        clear_btn.pack(side='left', padx=5)
        
        right_frame = tk.Frame(main_container, bg='#ecf0f1', width=400)
        right_frame.pack(side='right', fill='both', padx=5)
        right_frame.pack_propagate(False)
        
        self.result_frame = tk.LabelFrame(right_frame, text="📊 Prediction Results", 
                                         font=("Arial", 12, "bold"), bg='#ecf0f1', 
                                         fg='#2c3e50', padx=10, pady=10)
        self.result_frame.pack(fill='x', pady=5)
        
        self.result_label = tk.Label(self.result_frame, text="Enter patient data and click 'Analyze Risk'", 
                                     font=("Arial", 11), bg='#ecf0f1', fg='#7f8c8d', 
                                     wraplength=350, pady=20)
        self.result_label.pack()
        
        self.chart_frame = tk.LabelFrame(right_frame, text="📈 Risk Analysis", 
                                        font=("Arial", 12, "bold"), bg='#ecf0f1', 
                                        fg='#2c3e50', padx=10, pady=10)
        self.chart_frame.pack(fill='both', expand=True, pady=5)
        
        self.last_prediction = None
        self.last_probability = None
    
    def predict(self):
        try:
            height = float(self.entries['Height'].get())
            weight = float(self.entries['Weight'].get())
            bmi = weight / (height * height)
            
            input_values = []
            for feature in self.feature_names:
                if feature == 'BMI':
                    value = bmi
                elif feature in self.binary_fields:
                    value = 1 if self.entries[feature].get() == "Yes" else 0
                elif feature in self.dropdown_fields:
                    selection = self.entries[feature].get()
                    value = self.dropdown_fields[feature].index(selection) + 1 if feature != 'Sex' else self.dropdown_fields[feature].index(selection)
                else:
                    value = float(self.entries[feature].get())
                input_values.append(value)
            
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = self.scaler.transform(input_array)
            
            prediction = self.model.predict(input_scaled)
            probability = self.model.predict_proba(input_scaled)
            
            self.last_prediction = prediction[0]
            self.last_probability = probability[0]
            self.last_input_values = input_values
            
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            
            if prediction[0] == 1:
                risk_prob = probability[0][1] * 100
                
                result_box = tk.Frame(self.result_frame, bg='#fadbd8', relief='solid', bd=2)
                result_box.pack(fill='x', pady=5)
                
                tk.Label(result_box, text="⚠️", font=("Arial", 35), bg='#fadbd8', fg='#e74c3c').pack(pady=5)
                tk.Label(result_box, text="HIGH RISK OF DIABETES", font=("Arial", 14, "bold"), 
                        bg='#fadbd8', fg='#c0392b').pack()
                tk.Label(result_box, text=f"BMI: {bmi:.1f} | Risk: {risk_prob:.1f}%", font=("Arial", 11), 
                        bg='#fadbd8', fg='#922b21').pack(pady=5)
                tk.Label(result_box, text="Recommendation: Consult healthcare provider", 
                        font=("Arial", 9, "italic"), bg='#fadbd8', fg='#6e2c00', wraplength=350).pack(pady=5)
            else:
                healthy_prob = probability[0][0] * 100
                
                result_box = tk.Frame(self.result_frame, bg='#d5f4e6', relief='solid', bd=2)
                result_box.pack(fill='x', pady=5)
                
                tk.Label(result_box, text="✅", font=("Arial", 35), bg='#d5f4e6', fg='#27ae60').pack(pady=5)
                tk.Label(result_box, text="LOW RISK - Good Health", font=("Arial", 14, "bold"), 
                        bg='#d5f4e6', fg='#1e8449').pack()
                tk.Label(result_box, text=f"BMI: {bmi:.1f} | Confidence: {healthy_prob:.1f}%", font=("Arial", 11), 
                        bg='#d5f4e6', fg='#186a3b').pack(pady=5)
                tk.Label(result_box, text="Recommendation: Maintain healthy lifestyle", 
                        font=("Arial", 9, "italic"), bg='#d5f4e6', fg='#0e4f2e', wraplength=350).pack(pady=5)
            
            self.show_probability_chart()
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers!")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def show_probability_chart(self):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
        
        labels = ['Low Risk', 'High Risk']
        sizes = [self.last_probability[0] * 100, self.last_probability[1] * 100]
        colors = ['#27ae60', '#e74c3c']
        explode = (0.1, 0) if self.last_prediction == 0 else (0, 0.1)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax1.set_title('Risk Distribution')
        
        importances = None
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
        elif hasattr(self.model, 'estimators_'):
            avg_importances = np.zeros(len(self.feature_names))
            count = 0
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    avg_importances += estimator.feature_importances_
                    count += 1
            if count > 0:
                importances = pd.Series(avg_importances / count, index=self.feature_names)
        
        if importances is not None:
            top_5 = importances.nlargest(5)
            ax2.barh(range(len(top_5)), top_5.values, color='steelblue')
            ax2.set_yticks(range(len(top_5)))
            ax2.set_yticklabels([self.field_labels.get(f, f) for f in top_5.index], fontsize=8)
            ax2.set_xlabel('Importance', fontsize=8)
            ax2.set_title('Top 5 Risk Factors', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center')
            ax2.set_title('Feature Importance')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def save_record(self):
        if self.last_prediction is None:
            messagebox.showwarning("Warning", "Please make a prediction first!")
            return
        
        name = self.current_patient_name.get()
        if not name:
            messagebox.showwarning("Warning", "Please enter patient name!")
            return
        
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = ','.join(map(str, self.last_input_values))
        
        self.cursor.execute('''
            INSERT INTO patients (name, date, prediction, probability, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, date, int(self.last_prediction), float(self.last_probability[1]), data))
        self.conn.commit()
        
        messagebox.showinfo("Success", "Patient record saved successfully!")
    
    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Patient History")
        history_window.geometry("900x500")
        
        tree_frame = tk.Frame(history_window)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side='right', fill='y')
        
        tree = ttk.Treeview(tree_frame, yscrollcommand=scrollbar.set, 
                           columns=('ID', 'Name', 'Date', 'Risk', 'Probability'), show='headings')
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=tree.yview)
        
        tree.heading('ID', text='ID')
        tree.heading('Name', text='Patient Name')
        tree.heading('Date', text='Date')
        tree.heading('Risk', text='Risk Level')
        tree.heading('Probability', text='Risk %')
        
        tree.column('ID', width=50)
        tree.column('Name', width=200)
        tree.column('Date', width=150)
        tree.column('Risk', width=100)
        tree.column('Probability', width=100)
        
        self.cursor.execute('SELECT id, name, date, prediction, probability FROM patients ORDER BY date DESC')
        for row in self.cursor.fetchall():
            risk_text = "HIGH RISK" if row[3] == 1 else "LOW RISK"
            tree.insert('', 'end', values=(row[0], row[1], row[2], risk_text, f"{row[4]*100:.1f}%"))
        
        btn_frame = tk.Frame(history_window)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Delete Selected", command=lambda: self.delete_record(tree),
                 bg='#e74c3c', fg='white', padx=20, pady=5).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Close", command=history_window.destroy,
                 bg='#7f8c8d', fg='white', padx=20, pady=5).pack(side='left', padx=5)
    
    def delete_record(self, tree):
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a record to delete!")
            return
        
        if messagebox.askyesno("Confirm", "Delete selected record?"):
            for item in selected:
                record_id = tree.item(item)['values'][0]
                self.cursor.execute('DELETE FROM patients WHERE id = ?', (record_id,))
                tree.delete(item)
            self.conn.commit()
            messagebox.showinfo("Success", "Record deleted!")
    
    def batch_prediction(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        
        try:
            df = pd.read_csv(file_path)
            
            if not all(col in df.columns for col in self.feature_names):
                messagebox.showerror("Error", "CSV file must contain all required features!")
                return
            
            X = df[self.feature_names]
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            df['Prediction'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
            df['Risk_Probability'] = probabilities
            
            output_path = file_path.replace('.csv', '_predictions.csv')
            df.to_csv(output_path, index=False)
            
            messagebox.showinfo("Success", f"Batch prediction completed!\nSaved to: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch prediction failed: {str(e)}")
    
    def export_pdf(self):
        if self.last_prediction is None:
            messagebox.showwarning("Warning", "Please make a prediction first!")
            return
        
        default_name = f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
        
        try:
            c = pdf_canvas.Canvas(file_path, pagesize=letter)
            width, height = letter
            
            c.setFont("Helvetica-Bold", 20)
            c.drawString(inch, height - inch, "Diabetes Risk Assessment Report")
            
            c.setFont("Helvetica", 12)
            y = height - 1.5*inch
            
            c.drawString(inch, y, f"Patient Name: {self.current_patient_name.get()}")
            y -= 0.3*inch
            c.drawString(inch, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            y -= 0.3*inch
            
            risk_text = "HIGH RISK" if self.last_prediction == 1 else "LOW RISK"
            c.setFont("Helvetica-Bold", 14)
            c.drawString(inch, y, f"Prediction: {risk_text}")
            y -= 0.3*inch
            
            c.setFont("Helvetica", 12)
            prob = self.last_probability[1] * 100 if self.last_prediction == 1 else self.last_probability[0] * 100
            c.drawString(inch, y, f"Confidence: {prob:.1f}%")
            y -= 0.5*inch
            
            c.setFont("Helvetica-Bold", 14)
            c.drawString(inch, y, "Model Information:")
            y -= 0.3*inch
            
            c.setFont("Helvetica", 11)
            c.drawString(inch, y, f"Model Type: {self.model_info.get('best_model', 'N/A')}")
            y -= 0.25*inch
            c.drawString(inch, y, f"Model Accuracy: {self.model_info.get('accuracy', 0)*100:.2f}%")
            
            c.save()
            messagebox.showinfo("Success", f"PDF exported successfully!\n\nSaved to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"PDF export failed: {str(e)}")
    
    def show_model_comparison(self):
        if not hasattr(self, 'all_models') or not self.all_models:
            messagebox.showinfo("Info", "Model comparison data not available. Train advanced model first.")
            return
        
        comp_window = tk.Toplevel(self.root)
        comp_window.title("Model Comparison")
        comp_window.geometry("800x600")
        
        if os.path.exists('model_analysis.png'):
            from PIL import Image, ImageTk
            img = Image.open('model_analysis.png')
            img = img.resize((780, 580))
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(comp_window, image=photo)
            label.image = photo
            label.pack(fill='both', expand=True, padx=10, pady=10)
        else:
            tk.Label(comp_window, text="Model comparison chart not found.\nRun train_advanced_model.py first!",
                    font=("Arial", 12)).pack(expand=True)
    
    def show_feature_importance(self):
        imp_window = tk.Toplevel(self.root)
        imp_window.title("Feature Importance Analysis")
        imp_window.geometry("700x500")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        importances = None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
        elif hasattr(self.model, 'estimators_'):
            avg_importances = np.zeros(len(self.feature_names))
            count = 0
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    avg_importances += estimator.feature_importances_
                    count += 1
            if count > 0:
                importances = pd.Series(avg_importances / count, index=self.feature_names)
        
        if importances is not None:
            importances_sorted = importances.sort_values(ascending=True)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(importances_sorted)))
            importances_sorted.plot(kind='barh', ax=ax, color=colors)
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_title('Feature Importance for Diabetes Prediction', fontsize=14, fontweight='bold')
            ax.set_ylabel('Features', fontsize=12)
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=imp_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        else:
            tk.Label(imp_window, text="Feature importance not available for this model type.\n\nTry using Random Forest or Gradient Boosting model.",
                    font=("Arial", 12), wraplength=600, justify='center').pack(expand=True, pady=50)
            plt.close(fig)
    
    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        theme = self.colors['dark'] if self.dark_mode else self.colors['light']
        
        self.root.configure(bg=theme['bg'])
        
        for widget in self.root.winfo_children():
            self.apply_theme_recursive(widget, theme)
        
        mode_name = "Dark Mode" if self.dark_mode else "Light Mode"
        messagebox.showinfo("Theme Changed", f"{mode_name} activated! 🎨")
    
    def apply_theme_recursive(self, widget, theme):
        try:
            widget_type = widget.winfo_class()
            
            if widget_type in ['Frame', 'LabelFrame']:
                widget.configure(bg=theme['frame_bg'])
                if widget_type == 'LabelFrame':
                    widget.configure(fg=theme['label_fg'])
            elif widget_type == 'Label':
                if 'header' not in str(widget):
                    widget.configure(bg=theme['frame_bg'], fg=theme['label_fg'])
            elif widget_type == 'Button':
                pass
            elif widget_type == 'Radiobutton':
                widget.configure(bg=theme['frame_bg'], fg=theme['label_fg'], 
                               activebackground=theme['frame_bg'], selectcolor=theme['frame_bg'])
            
            for child in widget.winfo_children():
                self.apply_theme_recursive(child, theme)
        except:
            pass
    
    def show_about(self):
        about_text = f"""
Advanced Diabetes Prediction System
Version 2.0

Model: {self.model_info.get('best_model', 'N/A')}
Accuracy: {self.model_info.get('accuracy', 0)*100:.2f}%
Training Date: {self.model_info.get('training_date', 'N/A')}

Features:
✓ Multiple ML algorithms comparison
✓ Advanced hyperparameter tuning
✓ Patient records database
✓ Batch prediction from CSV
✓ PDF report generation
✓ Feature importance analysis
✓ Risk visualization charts

© 2025 Diabetes Prediction System
        """
        messagebox.showinfo("About", about_text)
    
    def clear_fields(self):
        for feature, widget in self.entries.items():
            if feature in self.binary_fields:
                widget.set("No")
            elif feature in self.dropdown_fields:
                widget.set(self.dropdown_fields[feature][0])
            elif feature == 'Height':
                widget.delete(0, tk.END)
                widget.insert(0, "1.70")
            elif feature == 'Weight':
                widget.delete(0, tk.END)
                widget.insert(0, "70")
            else:
                widget.delete(0, tk.END)
                widget.insert(0, "0")
        
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        self.current_patient_name.set("")
        self.last_prediction = None
        self.last_probability = None

def main():
    root = tk.Tk()
    app = AdvancedDiabetesPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
