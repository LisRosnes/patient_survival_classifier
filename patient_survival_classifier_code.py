import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import pearsonr

# Load your Excel sheet into a pandas dataframe
# Update the path as necessary
df = pd.read_excel('SurvivalPredictionTask.xlsx')

# Drop the SubjectID column
df = df.drop(columns=['SubjectID'])

#find median cuttoff for long vs short term survivors 
cut_off = df['Survival_from_surgery_days_UPDATED'].median()

# Defining a df distinguishing long vs short term survival based on cut_off
df['long_term_survival'] = np.where(df['Survival_from_surgery_days_UPDATED'] >= cut_off, 1, 0)

# My target and features
X = df.drop(columns=['Survival_from_surgery_days_UPDATED', 'long_term_survival'])
y = df['long_term_survival']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm = SVC(kernel='linear')

# Perform forward feature selection with up to 10 features
sfs = SequentialFeatureSelector(svm, n_features_to_select=10, direction='forward')
sfs.fit(X_train_scaled, y_train)

# Get the selected feature indices and their names
selected_features = sfs.get_support(indices=True)
selected_feature_names = X.columns[selected_features]

# Train the SVM model on the selected features
X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

svm.fit(X_train_selected, y_train)

# Predict on the test set
y_pred = svm.predict(X_test_selected)

# accuracy
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy of Classification: {acc}')

print("\nClassification Report:\n", classification_report(y_test, y_pred))
