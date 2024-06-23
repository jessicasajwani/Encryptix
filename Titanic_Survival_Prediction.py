import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Data
# You can download the Titanic dataset from Kaggle and save it as 'titanic.csv'
data = pd.read_csv('Titanic-Dataset.csv')

# Display first few rows of the data
print(data.head())

# Step 2: Data Preprocessing
# Remove rows with missing values for simplicity (you can also use imputation)
data.dropna(subset=['Age', 'Embarked'], inplace=True)

# Fill missing values in 'Cabin' with 'Unknown' (could also drop or impute differently)
data['Cabin'].fillna('Unknown', inplace=True)

# Fill missing 'Fare' values with median fare
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Select relevant features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Engineering
# One-hot encode categorical variables (Sex, Embarked)
# Standardize numerical features (Age, Fare, etc.)
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Step 4: Model Training
# Create a pipeline with the preprocessor and a classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Step 5: Evaluation
# Predict on the test set
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Display classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
