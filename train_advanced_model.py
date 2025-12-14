import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("=" * 60)
print("🚀 ADVANCED DIABETES PREDICTION MODEL TRAINING")
print("=" * 60)

df = pd.read_csv('big_diabetes.csv')
print(f"\n✅ Data Loaded! Rows: {df.shape[0]}, Columns: {df.shape[1]}")

target_col = 'Diabetes_binary'
if target_col not in df.columns:
    target_col = df.columns[0]

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("🧠 TRAINING MULTIPLE MODELS")
print("=" * 60)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42, early_stopping=True)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    start_time = datetime.now()
    
    model.fit(X_train_scaled, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"   ⏱️  Training time: {training_time:.1f} seconds")
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   Running cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    trained_models[name] = model
    
    print(f"   ✅ Accuracy: {accuracy*100:.2f}%")
    print(f"   ✅ CV Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

print("\n" + "=" * 60)
print("🎯 HYPERPARAMETER TUNING - Random Forest")
print("=" * 60)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("🔍 Searching for best parameters...")
grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"✅ Best CV Score: {grid_search.best_score_*100:.2f}%")

best_rf = grid_search.best_estimator_
trained_models['Tuned Random Forest'] = best_rf

y_pred_tuned = best_rf.predict(X_test_scaled)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
results['Tuned Random Forest'] = {
    'accuracy': tuned_accuracy,
    'cv_mean': grid_search.best_score_,
    'cv_std': 0
}

print("\n" + "=" * 60)
print("🤝 CREATING ENSEMBLE MODEL")
print("=" * 60)

ensemble = VotingClassifier(
    estimators=[
        ('rf', trained_models['Random Forest']),
        ('gb', trained_models['Gradient Boosting']),
        ('lr', trained_models['Logistic Regression'])
    ],
    voting='soft'
)

ensemble.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

trained_models['Ensemble'] = ensemble
results['Ensemble'] = {
    'accuracy': ensemble_accuracy,
    'cv_mean': ensemble_accuracy,
    'cv_std': 0
}

print(f"✅ Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")

print("\n" + "=" * 60)
print("📊 MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('accuracy', ascending=False)
print(results_df)

best_model_name = results_df.index[0]
best_model = trained_models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"🏆 Best Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")

print("\n" + "=" * 60)
print("📈 GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

accuracies = [results[m]['accuracy'] for m in results.keys()]
model_names = list(results.keys())
axes[0, 0].barh(model_names, accuracies, color='skyblue')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_title('Model Comparison - Accuracy')
axes[0, 0].set_xlim([0, 1])

y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')

if hasattr(best_model, 'predict_proba'):
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend(loc="lower right")

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(10)
    top_features.plot(kind='barh', ax=axes[1, 1], color='green')
    axes[1, 1].set_title('Top 10 Feature Importances')
    axes[1, 1].set_xlabel('Importance')
else:
    axes[1, 1].text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                   ha='center', va='center', fontsize=12)
    axes[1, 1].set_title('Feature Importances')

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization: model_analysis.png")

print("\n" + "=" * 60)
print("💾 SAVING MODELS AND DATA")
print("=" * 60)

joblib.dump(best_model, 'best_diabetes_model.pkl')
joblib.dump(scaler, 'advanced_scaler.pkl')
joblib.dump(trained_models, 'all_models.pkl')
joblib.dump(results, 'model_results.pkl')

model_info = {
    'best_model': best_model_name,
    'accuracy': results[best_model_name]['accuracy'],
    'feature_names': list(X.columns),
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'all_results': results
}
joblib.dump(model_info, 'model_info.pkl')

print("✅ Saved: best_diabetes_model.pkl")
print("✅ Saved: advanced_scaler.pkl")
print("✅ Saved: all_models.pkl")
print("✅ Saved: model_results.pkl")
print("✅ Saved: model_info.pkl")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nBest Model: {best_model_name}")
print(f"Final Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print("\nYou can now run the enhanced GUI!")
