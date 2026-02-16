##### Diabetes sensitivity analysis with ML and DL models #####

## Importing necessary libraries ##
# 1. Data manipulation and analysis
import pandas as pd
import numpy as np
import json

# 2. Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Data preprocessing and model selection
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from imblearn.over_sampling import SMOTE

# 4. Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 5. Deep Learning Models
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# 6. Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# 7. System and resource monitoring
import os
import time
import psutil
import warnings

warnings.filterwarnings('ignore')


### Main class for Diabetes ML Analysis ###

# This class encapsulates all functionalities for loading data, training models, evaluating performance, and analyzing resource usage. 
# It provides a structured way to perform comprehensive analysis on the diabetes dataset using both traditional machine learning and deep learning models. 
# The class includes methods for data preparation, model creation, fold evaluation, cross-validation, and visualization of results.
class DataMLAnalyzer:
    # Initialize the analyzer with dataset path and prepare storage for results and resource usage
    def __init__(self, data_path):
        self.data_path = data_path
        self.results = {}
        self.fold_results = {}
        self.resource_usage = {}
        self.confusion_matrices = {}
        self.training_histories = {}
        self.actual_training_scores = {}  # Store actual training performance
        self.actual_resource_metrics = {}  # Store actual resource usage
        
    # Load and prepare the dataset, including encoding, scaling, and balancing
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        
        print(f"Loading dataset from: {self.data_path}")
        data = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {data.shape}")
        print(f"Dataset columns: {list(data.columns)}")
        print(f"Class distribution:\n{data['CLASS'].value_counts()}")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            print(f"Missing values found:\n{missing_values[missing_values > 0]}")
        
        # Encode labels
        labels = data['CLASS']
        encoded_Y = LabelEncoder().fit_transform(labels)
        dummy_y = label_binarize(encoded_Y, classes=[0, 1, 2])
        
        # Features and scaling
        X = data.drop('CLASS', axis=1).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Original dataset shape: {X.shape}")
        print(f"Feature names: {list(data.drop('CLASS', axis=1).columns)}")
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42)
        X_res, y_res_labels = smote.fit_resample(X_scaled, np.argmax(dummy_y, axis=1))
        y_res = label_binarize(y_res_labels, classes=[0, 1, 2])
        
        print(f"After SMOTE shape: {X_res.shape}")
        print(f"Class distribution after SMOTE: {np.bincount(y_res_labels)}")
        
        return X_res, y_res, y_res_labels
    
    # Create deep learning models based on specified type (ANN, DNN, CNN, LSTM)
    def create_deep_model(self, model_type, input_shape, output_units):
        """Create deep learning models"""
        if model_type == 'ANN':
            model = models.Sequential([
                layers.Input(shape=(input_shape,)),
                layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(output_units, activation='softmax')
            ])
        elif model_type == 'DNN':
            model = models.Sequential([
                layers.Input(shape=(input_shape,)),
                layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                layers.Dropout(0.4),
                layers.Dense(128, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(output_units, activation='softmax')
            ])
        elif model_type == 'CNN':
            model = models.Sequential([
                layers.Conv1D(64, 3, activation='relu', input_shape=(input_shape, 1)),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(output_units, activation='softmax')
            ])
        elif model_type == 'LSTM':
            model = models.Sequential([
                layers.LSTM(64, input_shape=(input_shape, 1), return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(32),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(output_units, activation='softmax')
            ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Evaluate a single fold for a given model, including resource monitoring and performance metrics calculation
    def evaluate_model_fold(self, model, model_name, X_train, X_val, y_train, y_val, 
                          is_deep_learning=False, fold_num=0):
        """Evaluate model on a single fold measurements"""
        # Resource monitoring
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        if is_deep_learning:
            # Reshape data for CNN/LSTM if needed
            if model_name in ['CNN', 'LSTM']:
                X_train = X_train.reshape(-1, X_train.shape[1], 1)
                X_val = X_val.reshape(-1, X_val.shape[1], 1)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            # Fit and get training performance
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                              epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])
            
            # Store training history
            if model_name not in self.training_histories:
                self.training_histories[model_name] = []
            self.training_histories[model_name].append(history)
            
            # Get training performance
            train_pred_prob = model.predict(X_train, verbose=0)
            train_pred = np.argmax(train_pred_prob, axis=1)
            train_true = np.argmax(y_train, axis=1)
            
            # Validation predictions
            y_pred_prob = model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_val, axis=1)
        else:
            # Train the model and get training performance
            model.fit(X_train, np.argmax(y_train, axis=1))
            
            # training performance
            train_pred = model.predict(X_train)
            train_pred_prob = model.predict_proba(X_train)
            train_true = np.argmax(y_train, axis=1)
            
            # Validation predictions
            y_pred = model.predict(X_val)
            y_pred_prob = model.predict_proba(X_val)
            y_true = np.argmax(y_val, axis=1)
        
        # Resource monitoring end
        training_time = time.time() - start_time
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        cpu_after = process.cpu_percent()
        
        # validation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # training metrics
        train_accuracy = accuracy_score(train_true, train_pred)
        train_precision = precision_score(train_true, train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(train_true, train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(train_true, train_pred, average='weighted', zero_division=0)
        
        # Calculate specificity manually
        cm = confusion_matrix(y_true, y_pred)
        specificity_scores = []
        for i in range(len(cm)):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity_scores.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        specificity = np.mean(specificity_scores)
        
        if is_deep_learning:
            auc_roc = roc_auc_score(y_val, y_pred_prob, multi_class='ovr')
            train_auc_roc = roc_auc_score(y_train, train_pred_prob, multi_class='ovr')
        else:
            auc_roc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
            train_auc_roc = roc_auc_score(train_true, train_pred_prob, multi_class='ovr')
        
        # Store confusion matrix
        if model_name not in self.confusion_matrices:
            self.confusion_matrices[model_name] = []
        self.confusion_matrices[model_name].append(cm)
        
        return {
            # Validation metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc_roc': auc_roc,
            # training metrics
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_score': train_f1,
            'train_auc_roc': train_auc_roc,
            # resource metrics
            'training_time': training_time,
            'memory_used': memory_used,
            'cpu_usage': (cpu_before + cpu_after) / 2,
            'confusion_matrix': cm
        }
    
    # Run k-fold cross-validation for all models, including resource monitoring and performance aggregation
    def run_cross_validation(self, X, y, n_folds=5):
        """Run k-fold cross-validation for all models"""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Define models
        ml_models = [
            ("Logistic Regression", LogisticRegression(max_iter=1500)),
            ("Random Forest", RandomForestClassifier(n_estimators=20, max_depth=3, 
                                                   min_samples_split=3, max_features='sqrt', random_state=10)),
            ("Support Vector Machine", SVC(kernel='linear', probability=True, random_state=42)),
            ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=15, metric='manhattan')),
            ("Decision Tree", DecisionTreeClassifier(max_depth=15, min_samples_split=45, 
                                                   min_samples_leaf=45, max_features='sqrt', random_state=42))
        ]
        
        deep_models = ['ANN', 'DNN', 'CNN', 'LSTM']
        
        print(f"Starting {n_folds}-fold cross-validation ")
        
        # Evaluate ML models
        for model_name, model_class in ml_models:
            print(f"Evaluating {model_name}...")
            fold_results = []
            for fold_num, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = model_class
                results = self.evaluate_model_fold(model, model_name, X_train, X_val, 
                                                 y_train, y_val, is_deep_learning=False, fold_num=fold_num)
                fold_results.append(results)
            
            self.fold_results[model_name] = fold_results
            
            # Calculate averages from data
            self.results[model_name] = {
                'accuracy': np.mean([r['accuracy'] for r in fold_results]),
                'precision': np.mean([r['precision'] for r in fold_results]),
                'recall': np.mean([r['recall'] for r in fold_results]),
                'specificity': np.mean([r['specificity'] for r in fold_results]),
                'f1_score': np.mean([r['f1_score'] for r in fold_results]),
                'auc_roc': np.mean([r['auc_roc'] for r in fold_results]),
                'train_accuracy': np.mean([r['train_accuracy'] for r in fold_results]),
                'train_f1_score': np.mean([r['train_f1_score'] for r in fold_results]),
                'training_time': np.mean([r['training_time'] for r in fold_results]),
                'memory_used': np.mean([r['memory_used'] for r in fold_results]),
                'cpu_usage': np.mean([r['cpu_usage'] for r in fold_results])
            }
        
        # Evaluate Deep Learning models
        for model_name in deep_models:
            print(f"Evaluating {model_name}...")
            fold_results = []
            for fold_num, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = self.create_deep_model(model_name, X_train.shape[1], y_train.shape[1])
                results = self.evaluate_model_fold(model, model_name, X_train, X_val, 
                                                 y_train, y_val, is_deep_learning=True, fold_num=fold_num)
                fold_results.append(results)
            
            self.fold_results[model_name] = fold_results
            
            # Calculate averages from data
            self.results[model_name] = {
                'accuracy': np.mean([r['accuracy'] for r in fold_results]),
                'precision': np.mean([r['precision'] for r in fold_results]),
                'recall': np.mean([r['recall'] for r in fold_results]),
                'specificity': np.mean([r['specificity'] for r in fold_results]),
                'f1_score': np.mean([r['f1_score'] for r in fold_results]),
                'auc_roc': np.mean([r['auc_roc'] for r in fold_results]),
                'train_accuracy': np.mean([r['train_accuracy'] for r in fold_results]),
                'train_f1_score': np.mean([r['train_f1_score'] for r in fold_results]),
                'training_time': np.mean([r['training_time'] for r in fold_results]),
                'memory_used': np.mean([r['memory_used'] for r in fold_results]),
                'cpu_usage': np.mean([r['cpu_usage'] for r in fold_results])
            }
    
    # Create training vs testing performance comparison, including accuracy and F1-score
    def create_training_vs_testing_comparison(self):
        """Create training vs testing performance comparison"""
        models = list(self.results.keys())
        
        # Use training and testing performance
        training_acc = [self.results[model]['train_accuracy'] * 100 for model in models]
        testing_acc = [self.results[model]['accuracy'] * 100 for model in models]
        
        training_f1 = [self.results[model]['train_f1_score'] * 100 for model in models]
        testing_f1 = [self.results[model]['f1_score'] * 100 for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison 
        x = range(len(models))
        ax1.plot(x, training_acc, 'o-', label='Training', linewidth=2, markersize=8, color='blue')
        ax1.plot(x, testing_acc, 's-', label='Testing', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training vs Testing Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1-Score comparison
        ax2.plot(x, training_f1, 'o-', label='Training ', linewidth=2, markersize=8, color='blue')
        ax2.plot(x, testing_f1, 's-', label='Testing ', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('F1-Score (%)')
        ax2.set_title('Training vs Testing F1-Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
       
        plt.tight_layout()
        plt.savefig('Diabetes/results/training_vs_testing.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create resource usage analysis including training time, memory usage, and CPU usage for each model
    def create_resource_usage_analysis(self):
        """Create resource usage analysis """
        models = list(self.results.keys())
        training_times = [self.results[model]['training_time'] for model in models]
        memory_usage = [self.results[model]['memory_used'] for model in models]
        cpu_usage = [self.results[model]['cpu_usage'] for model in models]
        
        # Estimate testing time as a fraction of training time (realistic estimate)
        testing_times = [time * 0.05 for time in training_times]  # Testing typically much faster
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training vs Testing Time 
        x = range(len(models))
        width = 0.35
        ax1.bar([i - width/2 for i in x], training_times, width, label='Training Time ', alpha=0.8)
        ax1.bar([i + width/2 for i in x], testing_times, width, label='Testing Time (Estimated)', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Training vs Testing Time ')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Memory Usage 
        ax2.bar(models, memory_usage, color='skyblue', alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage ')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # CPU Usage
        ax3.bar(models, cpu_usage, color='lightcoral', alpha=0.8)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('CPU Usage (%)')
        ax3.set_title('CPU Usage ')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        # Show resource efficiency (Performance per resource)
        efficiency = [self.results[model]['accuracy'] / max(self.results[model]['training_time'], 0.001) 
                     for model in models]
        ax4.bar(models, efficiency, color='lightgreen', alpha=0.8)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Accuracy per Second')
        ax4.set_title('Resource Efficiency ')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Diabetes/results/resource_usage.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create performance comparison bar chart for all models across multiple metrics
    def create_performance_bar_chart(self):
        """Create performance comparison bar chart"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'recall', 'specificity', 'precision', 'f1_score', 'auc_roc']
        metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.13
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.results[model][metric] * 100 for model in models]
            ax.bar(x + i * width, values, width, label=label, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Models', fontweight='bold', fontsize=12)
        ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=16)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('Diabetes/results/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create confusion matrices visualization for all models using heatmaps
    def create_confusion_matrices_visualization(self):
        """Create confusion matrices visualization """
        n_models = len(self.confusion_matrices)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (model_name, cm_list) in enumerate(self.confusion_matrices.items()):
            # Use the confusion matrix from the last fold
            cm = cm_list[-1] if cm_list else np.zeros((3, 3))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('Diabetes/results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create training and validation loss curves for deep learning models
    def create_loss_curves(self):
        """Create training and validation loss curves """
        deep_models = [model for model in self.training_histories.keys()]
        if not deep_models:
            print("No deep learning models found for loss curves.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(deep_models[:4]):
            if idx >= len(axes):
                break
                
            histories = self.training_histories[model_name]
            
            for fold_num, history in enumerate(histories):
                # Plot training loss
                axes[idx].plot(history.history['loss'], 
                             label=f'Fold {fold_num+1} Train Loss', alpha=0.7)
                # Plot  validation loss
                axes[idx].plot(history.history['val_loss'], 
                             label=f'Fold {fold_num+1} Val Loss', alpha=0.7, linestyle='--')
            
            axes[idx].set_title(f'{model_name} -  Training and Validation Loss')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(deep_models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('Diabetes/results/loss_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Generate detailed performance tables for each fold and average performance for all models
    def generate_detailed_performance_tables(self):
        """Generate detailed performance tables"""
        tables = {}
        for model_name, fold_results in self.fold_results.items():
            data = []
            for i, fold_result in enumerate(fold_results):
                data.append({
                    'Fold': f'Fold {i+1}',
                    'Accuracy': f"{fold_result['accuracy']*100:.2f}%",
                    'Sensitivity': f"{fold_result['recall']*100:.2f}%",
                    'Specificity': f"{fold_result['specificity']*100:.2f}%",
                    'Precision': f"{fold_result['precision']*100:.2f}%",
                    'F1-Score': f"{fold_result['f1_score']*100:.2f}%",
                    'AUC-ROC': f"{fold_result['auc_roc']*100:.2f}%"
                })
            
            # Add average row 
            data.append({
                'Fold': 'Average',
                'Accuracy': f"{self.results[model_name]['accuracy']*100:.2f}%",
                'Sensitivity': f"{self.results[model_name]['recall']*100:.2f}%",
                'Specificity': f"{self.results[model_name]['specificity']*100:.2f}%",
                'Precision': f"{self.results[model_name]['precision']*100:.2f}%",
                'F1-Score': f"{self.results[model_name]['f1_score']*100:.2f}%",
                'AUC-ROC': f"{self.results[model_name]['auc_roc']*100:.2f}%"
            })
            
            tables[model_name] = pd.DataFrame(data)
        
        return tables
    
    # Generate summary performance table comparing all models across key metrics
    def generate_summary_table(self):
        """Generate summary performance table """
        data = []
        for model_name, results in self.results.items():
            data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']*100:.2f}%",
                'Sensitivity': f"{results['recall']*100:.2f}%",
                'Specificity': f"{results['specificity']*100:.2f}%",
                'Precision': f"{results['precision']*100:.2f}%",
                'F1-Score': f"{results['f1_score']*100:.2f}%",
                'AUC-ROC': f"{results['auc_roc']*100:.2f}%"
            })
        
        return pd.DataFrame(data)
    
    # Main method to run complete analysis including data loading, model training, performance evaluation, and visualization
    def run_complete_analysis(self):
        """Run complete analysis using """
        print("="*80)
        print("MACHINE LEARNING MODEL PERFORMANCE ANALYSIS")
        print("="*80)
        print(f"Using dataset: {self.data_path}")
       
        
        try:
            # Load and prepare data
            print("\nLoading and preparing data...")
            X, y, y_labels = self.load_and_prepare_data()
            
            # Run cross-validation
            print("\nRunning cross-validation with data...")
            self.run_cross_validation(X, y)
            
            print("\n" + "="*80)
            print("RESULTS")
            print("="*80)
            
            # 1. Detailed Performance Tables
            print("\n1. DETAILED PERFORMANCE TABLES ")
            print("-"*50)
            detailed_tables = self.generate_detailed_performance_tables()
            for model_name, table in detailed_tables.items():
                print(f"\n{model_name}")
                print(table.to_string(index=False))
            
            # 2. Summary Performance Table
            print("\n\n2. SUMMARY PERFORMANCE TABLE")
            print("-"*50)
            summary_table = self.generate_summary_table()
            print(summary_table.to_string(index=False))

            # Save result tables to files
            results_dir = os.path.join('Diabetes', 'results')
            os.makedirs(results_dir, exist_ok=True)
            try:
                # Save detailed tables per model
                for model_name, table in detailed_tables.items():
                    safe_name = model_name.replace(' ', '_')
                    table_path = os.path.join(results_dir, f"{safe_name}_detailed.csv")
                    table.to_csv(table_path, index=False)

                # Save summary table
                summary_path = os.path.join(results_dir, 'summary_table.csv')
                summary_table.to_csv(summary_path, index=False)

                # Save raw numerical results (convert numpy types to native Python floats)
                cleaned_results = {}
                for model, vals in self.results.items():
                    cleaned_results[model] = {k: float(v) for k, v in vals.items()}

                # Save confusion matrices and fold-level results as JSON-friendly structures
                cm_serializable = {m: [cm.astype(int).tolist() for cm in cms] for m, cms in self.confusion_matrices.items()}
                fold_serializable = {}
                for m, folds in self.fold_results.items():
                    fold_serializable[m] = [{k: (float(v) if isinstance(v, (np.floating, float, int)) else v) for k, v in fr.items() if k != 'confusion_matrix'} for fr in folds]

                aggregate = {
                    'results': cleaned_results,
                    'confusion_matrices': cm_serializable,
                    'fold_level': fold_serializable
                }

                with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
                    json.dump(aggregate, f, indent=2)

                print(f"Saved detailed tables to: {results_dir}")
                print(f"Saved summary table to: {summary_path}")
                print(f"Saved aggregated results to: {os.path.join(results_dir, 'results_summary.json')}")
            except Exception as e:
                print(f"Warning: failed to save result files: {e}")
            
            # 3. Generate visualizations
            print("\n\n3. GENERATING VISUALIZATIONS...")
            print("-"*50)
            
            print("Creating performance bar chart...")
            self.create_performance_bar_chart()
            
            print("Creating confusion matrices...")
            self.create_confusion_matrices_visualization()
            
            print("Creating training vs testing comparison...")
            self.create_training_vs_testing_comparison()
            
            print("Creating loss curves...")
            self.create_loss_curves()
            
            print("Creating resource usage analysis...")
            self.create_resource_usage_analysis()
            
            print("\n" + "="*80)
            print(" DATA ANALYSIS COMPLETE!")
            print("="*80)
            print("Generated files")
            print("- Diabetes/results/performance_comparison.png")
            print("- Diabetes/results/confusion_matrices.png") 
            print("- Diabetes/results/training_vs_testing.png")
            print("- Diabetes/results/loss_curves.png")
            print("- Diabetes/results/resource_usage.png")
            
            return {
                'detailed_tables': detailed_tables,
                'summary_table': summary_table,
                'results': self.results
            }
            
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print("Please ensure the dataset file exists at the specified path.")
            return None
        except Exception as e:
            print(f"\nERROR: An unexpected error occurred: {e}")
            return None

# Run the analysis when the script is executed directly
if __name__ == "__main__":
    # Set your dataset path here. Make sure the file exists at this location.
    data_path = 'Diabetes/diabetes_smote.csv'  # Your dataset path here
    analyzer = DataMLAnalyzer(data_path)
    
    
    # Run the complete analysis and get results
    results = analyzer.run_complete_analysis()
    
    # Provide final output message based on results
    if results:
        print("\nDATA analysis completed successfully!")
        print("You can access the results using:")
        print("- results['detailed_tables'] for individual model performance ")
        print("- results['summary_table'] for overall comparison")
        print("- results['results'] for raw numerical results")
    else:
        print("\nAnalysis failed. Please check the error messages above.")