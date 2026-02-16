import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from imblearn.over_sampling import SMOTE

# Load data and preprocessing
data = pd.read_csv('Diabetes/diabetes_cleaned_selected.csv')
labels = data['CLASS']
encoded_Y = LabelEncoder().fit_transform(labels)
dummy_y = label_binarize(encoded_Y, classes=[0, 1, 2])
X = data.drop('CLASS', axis=1).values
y = dummy_y
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate data after SMOTE
smote = SMOTE(random_state=42)
X_res, y_res_labels = smote.fit_resample(X_scaled, np.argmax(y, axis=1))

# Save SMOTE data as CSV
feature_names = data.drop('CLASS', axis=1).columns.tolist()
df_smote = pd.DataFrame(X_res, columns=feature_names)
df_smote['CLASS'] = y_res_labels
df_smote.to_csv('Diabetes/diabetes_smote.csv', index=False)
print(f"SMOTE data saved to 'Diabetes/diabetes_smote.csv'")
print(f"Shape: {df_smote.shape}")

# Original class distribution
original_counts = labels.value_counts().sort_index()

# SMOTE class distribution
smote_counts = pd.Series(y_res_labels).value_counts().sort_index()

# Combine both into a single DataFrame
df_counts = pd.DataFrame({
    'Original': original_counts,
    'After SMOTE': smote_counts
})

# Convert DataFrame to long format
df_long = df_counts.reset_index().melt(id_vars='index', var_name='Status', value_name='Count')
df_long.rename(columns={'index': 'Class'}, inplace=True)

plt.figure(figsize=(8,5))
sns.barplot(x='Class', y='Count', hue='Status', data=df_long, palette=['blue', 'orange'], alpha=0.8)
plt.title('Original and After SMOTE Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig('Diabetes/results/SMOTE_Before_After_diabetes.png')
plt.close()

