### Diabetes Prediction Code with Deep Learning and Machine Learning Models

import scipy.stats as stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, accuracy_score, roc_curve
from tensorflow.keras import layers, models, regularizers
from imblearn.over_sampling import SMOTE
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import joblib
import json

# Create results directory early
results_dir = 'pima/results'
os.makedirs(results_dir, exist_ok=True)

print("Pima Diabetes Dataset Analysis and Prediction")
file_path = 'pima/diabetes.csv'
data = pd.read_csv(file_path, encoding='latin1')

columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_to_check] = data[columns_to_check].replace(0, np.nan)

imputer = SimpleImputer(strategy='mean')
data[columns_to_check] = imputer.fit_transform(data[columns_to_check])

data['Output'] = data['Glucose'].apply(
    lambda x: 'diabetes' if x > 125 else 'prediabetes' if 99 < x <= 125 else 'normal' if x > 70 else 'diabetes'
)

label_map = {'normal': 0, 'prediabetes': 1, 'diabetes': 2}
data['Outcome'] = data['Output'].replace(label_map)
data = data.drop(columns=['Output'])

X_fs = data.drop(columns=['Outcome'])
y_fs = data['Outcome']
mi_scores = mutual_info_classif(X_fs, y_fs)
selected_features = list(X_fs.columns[np.argsort(mi_scores)[-5:]])

plt.figure(figsize=(10, 6))
plt.bar(selected_features, mi_scores[np.argsort(mi_scores)[-5:]])
plt.title('Selected Features based on Mutual Information')
plt.xticks(rotation=45)
plt.savefig(os.path.join(results_dir, 'mutual_info_selected_features.png'), bbox_inches='tight')
plt.close()

final_data = data[selected_features + ['Outcome']]
final_data.to_csv(os.path.join(results_dir, 'pima_cleaned_selected.csv'), index=False)

# Re-load for training
data = pd.read_csv(os.path.join(results_dir, 'pima_cleaned_selected.csv'))
labels = data['Outcome']
encoded_Y = LabelEncoder().fit_transform(labels)
dummy_y = label_binarize(encoded_Y, classes=[0, 1, 2])
X = data.drop('Outcome', axis=1).values
y = dummy_y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, np.argmax(y, axis=1))
y_res_bin = label_binarize(y_res, classes=[0, 1, 2])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Dictionary to hold raw fold results
detailed_dl_results = {}

def train_and_evaluate_model(model_type, train_features, train_labels, kf):
    fold_logs = []
    
    for fold_idx, (train_index, val_index) in enumerate(kf.split(train_features)):
        X_train, X_val = train_features[train_index], train_features[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]

        if model_type == 'ANN':
            model = models.Sequential([
                layers.Input(shape=(X_train.shape[1],)),
                layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(y_train.shape[1], activation='softmax')
            ])
        elif model_type == 'DNN':
            model = models.Sequential([
                layers.Input(shape=(X_train.shape[1],)),
                layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                layers.Dropout(0.4),
                layers.Dense(128, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(y_train.shape[1], activation='softmax')
            ])
        elif model_type == 'CNN':
            model = models.Sequential([
                layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(y_train.shape[1], activation='softmax')
            ])
        elif model_type == 'LSTM':
            model = models.Sequential([
                layers.LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True),
                layers.LSTM(32),
                layers.Dense(y_train.shape[1], activation='softmax')
            ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if model_type in ['CNN', 'LSTM']:
            X_train = X_train.reshape(-1, X_train.shape[1], 1)
            X_val = X_val.reshape(-1, X_val.shape[1], 1)

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])
        
        predictions = model.predict(X_val)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_val, axis=1)

        fold_logs.append({
            'fold': fold_idx + 1,
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='weighted'),
            'recall': recall_score(true_labels, pred_labels, average='weighted'),
            'auc_roc': roc_auc_score(y_val, predictions, multi_class='ovr')
        })

    detailed_dl_results[model_type] = fold_logs
    
    return {
        'Mean CV Accuracy': np.mean([f['accuracy'] for f in fold_logs]),
        'Mean CV Precision': np.mean([f['precision'] for f in fold_logs]),
        'Mean CV Recall': np.mean([f['recall'] for f in fold_logs]),
        'Mean CV AUC ROC': np.mean([f['auc_roc'] for f in fold_logs]),
    }

kf = KFold(n_splits=5, shuffle=True, random_state=42)
models_to_evaluate = ['ANN', 'DNN', 'CNN', 'LSTM']
deep_learning_summary = {}

for model_type in models_to_evaluate:
    print(f"Evaluating: {model_type}")
    deep_learning_summary[model_type] = train_and_evaluate_model(model_type, X_res, y_res_bin, kf)

ml_models = [
    ("Logistic Regression", LogisticRegression(max_iter=1500)),
    ("Random Forest", RandomForestClassifier(n_estimators=20, random_state=10)),
    ("Support Vector Machine", SVC(kernel='linear', probability=True, random_state=42)),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=15)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=15, random_state=42))
]

ml_summary = []
detailed_ml_results = {}

for model_name, model in ml_models:
    fold_logs = []
    for fold_idx, (train_index, val_index) in enumerate(kf.split(X_res)):
        X_train, X_val = X_res[train_index], X_res[val_index]
        y_train, y_val = y_res[train_index], y_res[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        fold_logs.append({
            'fold': fold_idx + 1,
            'accuracy': accuracy_score(y_val, y_pred),
            'auc_roc': roc_auc_score(y_val, y_proba, multi_class='ovr')
        })
    
    detailed_ml_results[model_name] = fold_logs
    ml_summary.append({
        "Model": model_name,
        "Mean CV Accuracy": np.mean([f['accuracy'] for f in fold_logs]),
        "Mean CV ROC AUC": np.mean([f['auc_roc'] for f in fold_logs])
    })

# --- SAVING SECTION ---

# 1. Save Summaries (CSV & JSON)
pd.DataFrame(ml_summary).to_csv(os.path.join(results_dir, 'ml_summary.csv'), index=False)
pd.DataFrame(deep_learning_summary).T.to_csv(os.path.join(results_dir, 'dl_summary.csv'))

with open(os.path.join(results_dir, 'all_results_summary.json'), 'w') as f:
    json.dump({"ML": ml_summary, "DL": deep_learning_summary}, f, indent=4)

# 2. Save Detailed Fold Results (CSV & JSON)
with open(os.path.join(results_dir, 'detailed_fold_results.json'), 'w') as f:
    json.dump({"ML_Folds": detailed_ml_results, "DL_Folds": detailed_dl_results}, f, indent=4)

# Convert nested fold dicts to flat CSV for easier Excel reading
dl_folds_df = pd.concat({k: pd.DataFrame(v) for k, v in detailed_dl_results.items()}, axis=0)
dl_folds_df.to_csv(os.path.join(results_dir, 'detailed_dl_folds.csv'))

# 3. Visualizations (PNG)
ml_results_df = pd.DataFrame(ml_summary)
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Mean CV Accuracy", data=ml_results_df, palette="viridis")
plt.title("ML Model Accuracy Comparison")
plt.savefig(os.path.join(results_dir, 'ml_accuracy_comparison.png'))
plt.close()

print(f"All files (JSON, CSV, PNG) have been saved successfully to: {results_dir}")