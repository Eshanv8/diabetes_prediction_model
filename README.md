# 🏥 Advanced Diabetes Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive machine learning system for diabetes risk prediction featuring multiple ML algorithms, advanced GUI, patient database, and analytics tools.

![Project Banner](https://img.shields.io/badge/Accuracy-86.58%25-success)
![Models](https://img.shields.io/badge/Models-6-blue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📸 Screenshots

### Advanced GUI Interface
- User-friendly input forms with radio buttons and spinboxes
- Real-time risk analysis with probability charts
- Feature importance visualization
- Patient history management

### Model Comparison Dashboard
- ROC curves and confusion matrices
- Accuracy comparison across 6 different models
- Feature importance rankings

---

## 🎯 Features

### 🤖 Machine Learning
- **6 ML Algorithms**: Random Forest, Gradient Boosting, Logistic Regression, Neural Network, Tuned RF, Ensemble
- **86.58% Accuracy**: Achieved with Ensemble model
- **Hyperparameter Tuning**: Grid Search optimization
- **Cross-Validation**: 3-fold CV for robust evaluation

### 🖥️ Interactive GUI
- **Real-time Predictions**: Instant diabetes risk assessment
- **Visual Analytics**: Pie charts, bar charts, ROC curves
- **Patient Database**: SQLite-based record management
- **PDF Reports**: Professional exportable reports
- **Batch Processing**: Upload CSV for multiple predictions

### 📊 Advanced Analytics
- Feature importance analysis
- Model comparison visualization
- ROC curves and confusion matrices
- Risk probability distribution

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction-system.git
cd diabetes-prediction-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Place `big_diabetes.csv` in the project root
- Dataset: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

### Usage

**Step 1: Train Models**
```bash
python train_advanced_model.py
```
*This trains 6 models, performs hyperparameter tuning, and saves the best model (~1-2 minutes)*

**Step 2: Launch GUI**
```bash
python gui_advanced.py
```

---

## 📁 Project Structure

```
diabetes-prediction-system/
├── train_advanced_model.py    # Advanced model training
├── gui_advanced.py            # Enhanced GUI application
├── train_big_model.py         # Basic model training
├── gui_app.py                 # Basic GUI
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── .gitignore                # Git ignore rules
└── data/
    └── big_diabetes.csv      # Dataset (not included)
```

---

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **GUI**: Tkinter
- **Visualization**: Matplotlib, Seaborn
- **Database**: SQLite3
- **PDF Generation**: ReportLab
- **Data Processing**: Pandas, NumPy

---

## 📊 Model Performance

| Model | Accuracy | CV Score |
|-------|----------|----------|
| **Ensemble** | **86.58%** | **86.58%** |
| Gradient Boosting | 86.54% | 86.70% |
| Tuned Random Forest | 86.53% | 86.61% |
| Neural Network | 86.49% | 86.57% |
| Random Forest | 86.41% | 86.48% |
| Logistic Regression | 86.21% | 86.38% |

---

## 💡 Key Features Explained

### Multi-Model Training
Trains and compares 6 different algorithms to find the best performer:
- Random Forest with default settings
- Hyperparameter-tuned Random Forest
- Gradient Boosting
- Logistic Regression
- Neural Network (MLP)
- Ensemble (voting classifier)

### Patient Record Management
- Save patient predictions to SQLite database
- View complete patient history
- Delete old records
- Export individual reports to PDF

### Batch Prediction
- Upload CSV files with multiple patient records
- Get predictions for all patients at once
- Export results to new CSV file

### Visual Analytics
- Real-time probability distribution charts
- Top 5 risk factors for each prediction
- Complete feature importance ranking
- Model comparison dashboard

---

## 🎓 Input Fields Guide

### Binary Fields (Yes/No)
- High Blood Pressure, High Cholesterol, Smoker, Stroke History
- Heart Disease, Physical Activity, Fruits/Veggies Consumption
- Healthcare Access, Walking Difficulty

### Numeric Fields
- **BMI**: 15-50 (Body Mass Index)
- **General Health**: 1=Excellent to 5=Poor
- **Mental/Physical Health**: Days per month (0-30)
- **Age Category**: 1-13 (18-24 to 80+)
- **Education**: 1-6 (Never attended to College graduate)
- **Income**: 1-8 (<$10k to >$75k)

---

## 📈 Future Enhancements

- [ ] Deep Learning models (LSTM, Transformer)
- [ ] SHAP values for explainability
- [ ] Web-based interface (Flask/Streamlit)
- [ ] RESTful API
- [ ] Mobile app version
- [ ] Real-time data integration
- [ ] Multi-disease prediction

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a medical device and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

## 📞 Contact

**Your Name** - [Your LinkedIn](https://linkedin.com/in/your-profile)

Project Link: [https://github.com/YOUR_USERNAME/diabetes-prediction-system](https://github.com/YOUR_USERNAME/diabetes-prediction-system)

---

## 🙏 Acknowledgments

- Dataset: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- scikit-learn documentation
- Tkinter GUI framework
- Python community

---

**⭐ Star this repository if you found it helpful!**

---

## 🚀 Quick Start Guide

### Step 1: Train the Advanced Model
Run this first to train all models and create comparison charts:

```bash
python train_advanced_model.py
```

**What it does:**
- ✓ Trains 5 different ML algorithms (Random Forest, Gradient Boosting, Logistic Regression, SVM, Neural Network)
- ✓ Performs hyperparameter tuning
- ✓ Creates ensemble model
- ✓ Compares all models and selects the best one
- ✓ Generates visualization charts (ROC curve, confusion matrix, feature importance)
- ✓ Saves all models and results

**Output files:**
- `best_diabetes_model.pkl` - Best performing model
- `advanced_scaler.pkl` - Data scaler
- `all_models.pkl` - All trained models
- `model_results.pkl` - Performance metrics
- `model_info.pkl` - Model metadata
- `model_analysis.png` - Visual comparison charts

---

### Step 2: Run the Advanced GUI
Launch the enhanced graphical interface:

```bash
python gui_advanced.py
```

---

## ✨ Features

### 🤖 Machine Learning
1. **Multiple Algorithms**
   - Random Forest
   - Gradient Boosting
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Neural Network (MLP)
   - Ensemble (Voting Classifier)

2. **Advanced Training**
   - Hyperparameter tuning with Grid Search
   - Cross-validation
   - Model comparison
   - Automatic best model selection

3. **Evaluation Metrics**
   - Accuracy scores
   - ROC curves and AUC
   - Confusion matrix
   - Classification reports

### 🖥️ GUI Features

#### 📊 Main Interface
- **User-friendly Input**: Radio buttons for Yes/No questions, spinboxes for numeric values
- **Real-time Predictions**: Instant risk assessment with probability
- **Visual Feedback**: Color-coded results (green=low risk, red=high risk)
- **Patient Name Entry**: Track individual patients

#### 📈 Analytics & Visualization
- **Probability Charts**: Pie chart showing risk distribution
- **Feature Importance**: Bar chart of top 5 risk factors
- **Model Comparison**: View performance of all algorithms
- **Full Feature Analysis**: Complete feature importance ranking

#### 💾 Data Management
- **SQLite Database**: Automatic storage of patient records
- **Patient History**: View all past predictions
- **Record Management**: Save, view, and delete patient records
- **Search & Filter**: Easy access to historical data

#### 📄 Export Features
- **PDF Reports**: Generate professional PDF reports with:
  - Patient information
  - Prediction results
  - Model details
  - Timestamp
- **Batch Prediction**: Upload CSV files for multiple predictions
- **CSV Export**: Batch results saved to CSV

#### 🎨 Interface Elements
- **Menu System**:
  - File → Batch Prediction, Export PDF, Exit
  - View → Patient History, Model Comparison, Feature Importance, Dark Mode
  - Help → About
- **Modern Design**: Professional color scheme and layout
- **Responsive**: Scrollable forms for all screen sizes

---

## 📁 File Structure

```
ML/
├── diabetes.csv                    # Original small dataset
├── big_diabetes.csv               # Large dataset (200K+ records)
├── model.py                       # Basic training script
├── train_big_model.py            # Large dataset training
├── train_advanced_model.py       # ⭐ Advanced multi-model training
├── gui_app.py                    # Basic GUI
├── gui_advanced.py               # ⭐ Advanced GUI with all features
├── patient_records.db            # SQLite database (auto-created)
├── model_analysis.png            # Model comparison charts
└── Generated Models:
    ├── best_diabetes_model.pkl
    ├── advanced_scaler.pkl
    ├── all_models.pkl
    ├── model_results.pkl
    └── model_info.pkl
```

---

## 🎯 Usage Guide

### Making a Prediction

1. **Enter Patient Information**
   - Answer Yes/No for binary questions (smoking, high BP, etc.)
   - Enter numeric values for BMI, age category, etc.

2. **Enter Patient Name** (optional but recommended for record keeping)

3. **Click "Analyze Risk"**
   - View prediction result
   - See probability distribution
   - Check top risk factors

4. **Save Record** (optional)
   - Click "Save Record" to store in database

### Viewing Patient History

1. Go to **View → Patient History**
2. See all past predictions with:
   - Patient name
   - Date and time
   - Risk level
   - Probability percentage
3. Delete records if needed

### Batch Prediction

1. Prepare a CSV file with all required columns:
   - Must have exact column names matching the model
   - One row per patient

2. Go to **File → Batch Prediction**

3. Select your CSV file

4. Results saved to `filename_predictions.csv`

### Exporting PDF Report

1. Make a prediction first
2. Go to **File → Export to PDF**
3. Choose save location
4. Professional PDF report generated

### Viewing Model Comparison

1. Go to **View → Model Comparison**
2. See charts comparing:
   - Accuracy of all models
   - Confusion matrix
   - ROC curve
   - Feature importance

---

## 📊 Understanding the Results

### Risk Levels
- **HIGH RISK**: Model predicts diabetes (>50% probability)
  - Red color indicator
  - Recommendation: Consult healthcare provider

- **LOW RISK**: Model predicts no diabetes (<50% probability)
  - Green color indicator
  - Recommendation: Maintain healthy lifestyle

### Probability
- Percentage confidence in the prediction
- Higher percentage = more confident prediction

### Feature Importance
- Shows which factors matter most for predictions
- Typical top factors:
  - BMI (Body Mass Index)
  - Age
  - General Health
  - High Blood Pressure
  - Glucose levels

---

## 🛠️ Technical Details

### Models Used
1. **Random Forest**: Ensemble of decision trees
2. **Gradient Boosting**: Sequential tree building
3. **Logistic Regression**: Linear probability model
4. **SVM**: Support vector classification
5. **Neural Network**: Multi-layer perceptron
6. **Ensemble**: Combines top 3 models via voting

### Hyperparameter Tuning
- Grid Search with 3-fold cross-validation
- Parameters optimized:
  - Number of trees
  - Tree depth
  - Minimum samples for split/leaf

### Data Processing
- StandardScaler normalization
- Train/test split: 80/20
- Stratified sampling (maintains class balance)

---

## 📦 Dependencies

All installed automatically:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- reportlab (PDF generation)
- pillow (image handling)
- tkinter (built-in with Python)
- sqlite3 (built-in with Python)

---

## 🔧 Troubleshooting

### Model Not Found Error
**Solution**: Run `train_advanced_model.py` first to create the models

### CSV Format Error (Batch Prediction)
**Solution**: Ensure CSV has exact column names:
```
HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income
```

### GUI Not Opening
**Solution**: Check if all packages installed:
```bash
pip list
```

---

## 📈 Model Performance

After training, typical results:
- **Accuracy**: 75-78%
- **Best Model**: Usually Random Forest or Gradient Boosting
- **Cross-Validation Score**: 74-77%

---

## 🎓 Field Guide

### Binary Fields (Yes/No)
- High Blood Pressure
- High Cholesterol
- Cholesterol Check (last 5 years)
- Smoker
- History of Stroke
- Heart Disease or Attack
- Physical Activity
- Eat Fruits Daily
- Eat Vegetables Daily
- Heavy Alcohol Consumption
- Have Health Insurance
- Difficulty Walking
- Sex (0=Female, 1=Male)

### Numeric Fields
- **BMI**: Body Mass Index (15-50 typical range)
- **General Health**: 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor
- **Mental Health**: Days with mental health issues (0-30)
- **Physical Health**: Days with physical health issues (0-30)
- **Age**: Category 1-13 (1=18-24, 13=80+)
- **Education**: Level 1-6 (1=Never, 6=College graduate)
- **Income**: Level 1-8 (1=<$10k, 8=>$75k)

---

## 🚀 Future Enhancements

Potential additions:
- Deep learning models (LSTM, CNN)
- SHAP values for better explainability
- Web-based interface (Flask/Django)
- Real-time data integration
- Mobile app version
- Multi-language support
- Prescription recommendations
- Diet and exercise suggestions

---

## 📝 Notes

- **Not a Medical Device**: This is for educational/research purposes only
- **Consult Professionals**: Always consult healthcare providers for medical decisions
- **Data Privacy**: Patient records stored locally in SQLite database
- **Backup**: Regularly backup `patient_records.db` file

---

## 🎉 Credits

Developed using:
- Scikit-learn ML library
- Tkinter GUI framework
- Matplotlib/Seaborn visualization
- ReportLab PDF generation

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review error messages
3. Verify all dependencies installed
4. Ensure training completed successfully

---

**Enjoy your advanced diabetes prediction system!** 🏥✨
