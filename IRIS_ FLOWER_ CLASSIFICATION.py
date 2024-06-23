import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = pd.read_csv('IRIS.csv')

# Preprocess the data
# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X = scaler.fit_transform(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
y = iris['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Use the model to make predictions on new, unseen data
new_flower = pd.DataFrame({'sepal_length': [5.1], 
                            'sepal_width': [3.5], 
                            'petal_length': [1.4], 
                            'petal_width': [0.2]})
new_flower_features = scaler.transform(new_flower)
predicted_species = model.predict(new_flower_features)
print(f'Predicted Species: {predicted_species[0]}')