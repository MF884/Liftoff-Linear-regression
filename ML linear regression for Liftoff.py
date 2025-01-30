import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()

missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing Values:\n", missing_values)

sns.histplot(train['SalePrice'], kde=True, bins=30)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.show()

numeric_cols = train.select_dtypes(include=['number'])

# Calculate correlations
correlation = numeric_cols.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=False, cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# Extract target variable
y_train = train['SalePrice']

# Select only numerical features
train = train.select_dtypes(include=[np.number])
train = train.select_dtypes(include=[np.number]).fillna(train.mean())
test = test.select_dtypes(include=[np.number])
test = test.select_dtypes(include=[np.number]).fillna(test.mean())


# Define features (excluding target variable)
X_train = train.drop(columns=['SalePrice'])

# Align train and test datasets
test = test[X_train.columns]  # Ensures test only has relevant features

# Verify final shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {test.shape}")

# Split the training data
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_split, y_train_split)

# Validate the model
y_val_pred = model.predict(X_val)

# Calculate performance metrics
mae = mean_absolute_error(y_val, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

# Print results
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)
