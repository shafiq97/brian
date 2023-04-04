import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset from the 'dataset' sheet in the 'dataset.xlsx' file
df = pd.read_excel('dataset.xlsx', sheet_name='dataset')

# Split the dataset into separate sheets for the unhealthy and healthy data
unhealthy_df = pd.read_excel('dataset.xlsx', sheet_name='Unhealthy')
healthy_df = pd.read_excel('dataset.xlsx', sheet_name='Healthy')

# Split the data into training and testing sets
train_df = pd.read_excel('dataset.xlsx', sheet_name='Training')
test_df = pd.read_excel('dataset.xlsx', sheet_name='Test')

# Extract the 'Classification' column as the target variable
y = df['Classification']

# Extract the 2nd and 3rd columns as the features
X = df.iloc[:, 1:3]

# Create and train a random forest classifier
rfc = RandomForestClassifier()
rfc.fit(X, y)

# Use the trained classifier to make predictions on the testing data
X_test = test_df.iloc[:, 1:3]
y_pred = rfc.predict(X_test)

# Compute the accuracy of the classifier on the testing data
y_true = test_df.iloc[:, 0]
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Create a scatter plot of the healthy and unhealthy data
plt.scatter(unhealthy_df.iloc[:, 1], unhealthy_df.iloc[:, 2], color='red', label='Unhealthy')
plt.scatter(healthy_df.iloc[:, 1], healthy_df.iloc[:, 2], color='green', label='Healthy')

# Create a scatter plot of the testing data with the predicted labels
plt.scatter(X_test[y_pred == 0].iloc[:, 0], X_test[y_pred == 0].iloc[:, 1], color='yellow', label='Predicted Healthy')
plt.scatter(X_test[y_pred == 1].iloc[:, 0], X_test[y_pred == 1].iloc[:, 1], color='blue', label='Predicted Unhealthy')

plt.xlabel('Feature DC')
plt.ylabel('Feature IR')
plt.title('Random Forest Classifier')
plt.legend()
plt.show()
