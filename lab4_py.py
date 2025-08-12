
#A1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("MC.csv")

# Drop columns that are not useful for prediction
df = df.drop(columns=['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue'])

# Handle categorical variables
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['Response'])
y = df['Response']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred, dataset_name):
    print(f"\n--- {dataset_name} Data ---")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(classification_report(y_true, y_pred))

# Evaluate on both train and test data
evaluate_model(y_train, y_train_pred, "Training")
evaluate_model(y_test, y_test_pred, "Test")

# Fit outcome
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

if abs(train_acc - test_acc) < 0.05:
    print("Model is Regularfit ✅")
elif train_acc > test_acc:
    print("Model is Overfit ⚠️")
else:
    print("Model is Underfit ⚠️")

#A4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from datetime import datetime


def generateData(n_features, n_classes, step, n_data_pts, lower_limit, upper_limit):
  possible_values = np.array(np.arange(lower_limit, upper_limit + step, step))
  possible_class_labels = np.arange(n_classes)
  data = {}
  for i in range(n_features):
    data[f"X{i+1}"] = np.random.choice(possible_values, n_data_pts)
  data["target"] = np.random.choice(possible_class_labels, n_data_pts)

  df = pd.DataFrame(data)

  return df

def plotData(feature1, feature2, target, color_map, title):
  plt.scatter(feature1, feature2, c= [color_map[pt] for pt in target])
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.title(title)
  plt.show()

def knnModel(X_train, X_test, y_train, y_test, k):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  predictions = knn.predict(X_test)
  return predictions

def modelTuning(X_train, y_train, X_test, y_test):
  knn = KNeighborsClassifier()
  params = {
    "n_neighbors": np.arange(1,int(len(X_train)**0.5), 2),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"]
  }
  rscv = RandomizedSearchCV(knn, params, cv=5)
  rscv.fit(X_train, y_train)
  return rscv.best_params_, rscv.best_score_
#A3
if __name__ == "__main__":
  train  = generateData(2, 2, 1, 20, 1, 10)
  plotData(train["X1"], train["X2"], train["target"],{0: "blue", 1: "red"}, "A3")

test = generateData(2, 2, 0.1, 10000, 1, 10)
predictions = knnModel(train[["X1", "X2"]], test[["X1", "X2"]], train["target"], test["target"], 3)
plotData(test["X1"], test["X2"], predictions, {0: "blue", 1: "red"}, "Classification when K = 3")

#A5
for i in range(1,21,2):
  predictions = knnModel(train[["X1", "X2"]], test[["X1", "X2"]], train["target"], test["target"], i)
  plotData(test["X1"], test["X2"], predictions, {0: "blue", 1: "red"}, f"Classification when K = {i}")

#A6
# reading the data
df = pd.read_excel("Lab Session Data.xlsx", sheet_name = "marketing_campaign")
print(df.head(2))

# data preprocessing
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], dtype=int)
df["Age"] = datetime.now().year - df["Year_Birth"]
df = df.drop(columns=['Year_Birth', 'Dt_Customer'])

print("Null values")
print(df.isnull().sum())

df = df.fillna(df.mean())
print("null values")
print(df.isnull().sum())

# splitting the data for model training

X = df[["Income", "Recency"]]
# target variable for 2 chosen classes with label encoding
y = df["Response"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  # observing class boundaries for different k values
print("Class boundaries for different k values for marketing campaign")
for i in range(1,21, 2):
  predictions = knnModel(X_train, X_test, y_train, y_test, i)
  plotData(X_test["Income"], X_test["Recency"], predictions, {0: "blue", 1: "red"}, f"Classification when K = {i}")

# A7
print("Best parameters and score for the synthetic data generated",modelTuning(train[["X1", "X2"]], train["target"], test[["X1", "X2"]], test["target"]))
print("Best parameters and score for the marketing campaign data",modelTuning(X_train, y_train, X_test, y_test))

#A2
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

def load_and_clean_data(file_path):
    """
    Loads the CSV file, removes unwanted characters from numeric columns,
    and returns a cleaned DataFrame.
    """
    df = pd.read_csv(file_path).dropna(how='all')

    # Clean numeric columns: remove commas and convert to float
    for column in ['Price', 'Open', 'High', 'Low']:
        df[column] = df[column].str.replace(",", "").astype(float)

    # Clean percentage column: remove '%' and convert to float
    df['Chg%'] = df['Chg%'].str.replace('%', '').astype(float)

    # Clean categorical text columns
    df['Day'] = df['Day'].str.strip()
    df['Month'] = df['Month'].str.strip()

    return df

def train_regression_model(features, target):
    """
    Trains a Linear Regression model on the given features and target.
    Returns the trained model and predicted values.
    """
    model = LinearRegression()
    model.fit(features, target)
    predictions = model.predict(features)
    return model, predictions

def calculate_regression_metrics(actual_values, predicted_values):
    """
    Calculates and returns MSE, RMSE, MAPE, and R-squared for model evaluation.
    """
    mse_score = mean_squared_error(actual_values, predicted_values)
    rmse_score = np.sqrt(mse_score)
    mape_score = mean_absolute_percentage_error(actual_values, predicted_values)
    r2_score_value = r2_score(actual_values, predicted_values)

    return mse_score, rmse_score, mape_score, r2_score_value

# Main Program
if __name__ == "__main__":
    # Load and preprocess the data
    file_path = "112.csv"  # Ensure this file is in the same directory
    stock_data = load_and_clean_data(file_path)

    # Select input features and target variable
    input_features = stock_data[['Open', 'High', 'Low']]
    target_price = stock_data['Price']

    # Train model and get predictions
    model, predicted_price = train_regression_model(input_features, target_price)

    # Evaluate the model
    mse, rmse, mape, r2 = calculate_regression_metrics(target_price, predicted_price)

    # Display the evaluation results
    print("📊 Model Evaluation Metrics:")
    print(f"  MSE  (Mean Squared Error):      {mse:.2f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"  MAPE (Mean Absolute % Error):   {mape * 100:.2f}%")
    print(f"  R² Score (R-squared):           {r2:.4f}")
