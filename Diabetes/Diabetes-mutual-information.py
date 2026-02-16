# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif

# 1. Load the Dataset
file_path = 'Diabetes/Dataset of Diabetes .csv'
data = pd.read_csv(file_path, encoding='latin1')

print("1. First 5 rows of the original dataset:")
print(data.head())

# 2. Handle Missing Values – Replace 0 with NaN in specific columns
columns_to_check = ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
data[columns_to_check] = data[columns_to_check].replace(0, np.nan)

print("\n2.1 - After replacing 0s with NaN:")
print(data.head())

# Fill missing values with median
imputer = SimpleImputer(strategy='median')
data[columns_to_check] = imputer.fit_transform(data[columns_to_check])

print("\n2.2 - After filling missing values with median:")
print(data.head())

# Filter out invalid values in the CLASS column
valid_classes = ['N', 'Y', 'P']
data = data[data['CLASS'].isin(valid_classes)]

print("\n2.3 - After filtering valid values in the CLASS column:")
print(data.head())

# Convert CLASS labels to numeric
class_mapping = {'N': 0, 'Y': 1, 'P': 2}
data['CLASS'] = data['CLASS'].map(class_mapping)

print("\n2.4 - After converting CLASS labels to numeric:")
print(data.head())

# Check unique values in Gender column
print("Unique values in the Gender column:", data['Gender'].unique())

# Remove invalid values from Gender column
data = data[data['Gender'].isin(['F', 'M'])]

print("\n2.5 - After filtering valid values in the Gender column:")
print(data.head())

# Convert Gender to numeric
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})

print("\n2.6 - After converting Gender to numeric:")
print(data.head())

# Final check for missing values
assert data['CLASS'].isna().sum() == 0, "There are still NaN values in the CLASS column!"
assert data['Gender'].isna().sum() == 0, "There are still missing values in the Gender column!"

print("\n2.7 - Final cleaned dataset preview:")
print(data.head())

# 3. Feature Selection using Mutual Information
X = data.drop(columns=['CLASS'])
y = data['CLASS']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
mi_scores_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
ax = sns.barplot(x=mi_scores_series.values, y=mi_scores_series.index, palette='viridis')
if ax.legend_: ax.legend_.remove()
plt.title('Mutual Information Scores of All Features')
plt.xlabel('Mutual Information Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('Diabetes/results/mutual_information_scores_diabetes.png', dpi=300)
plt.show()

# 5. Print and select the top 5 features with highest MI scores
selected_features = mi_scores_series.head(5).index.tolist()
print("\n5. Top 5 features with highest Mutual Information scores:")
for feature in selected_features:
    print(f"{feature}: {mi_scores_series[feature]:.4f}")

# 6. Create a new dataset with selected features and save it
final_data = data[selected_features + ['CLASS']]

print("\n6.1 - First 5 rows of the new dataset with selected features and CLASS:")
print(final_data.head())

final_data.to_csv('Diabetes/diabetes_cleaned_selected.csv', index=False)
print("\n6.2 - The dataset has been successfully cleaned and saved with selected features.")
