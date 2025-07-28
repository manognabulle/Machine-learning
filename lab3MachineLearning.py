import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from scipy.spatial.distance import minkowski

# --- Data Loading & Preprocessing ---
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['ID', 'Dt_Customer', 'Education', 'Marital_Status'])
    df = df.dropna()
    df = df[df['Response'].isin([0, 1])]  # Binary classification only
    X = df.drop(columns=['Response']).values
    y = df['Response'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# --- A1: Intraclass spread and interclass distance ---
def analyze_classes(X, y):
    class0 = X[y == 0]
    class1 = X[y == 1]
    centroid0 = np.mean(class0, axis=0)
    centroid1 = np.mean(class1, axis=0)
    std0 = np.std(class0, axis=0)
    std1 = np.std(class1, axis=0)
    interclass_dist = np.linalg.norm(centroid0 - centroid1)
    return centroid0, centroid1, std0, std1, interclass_dist

# --- A2: Histogram analysis for one feature ---
def feature_histogram(X, feature_index=0):
    data = X[:, feature_index]
    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.title("A2: Histogram of Feature {}".format(feature_index))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("A2_histogram.png")
    plt.close()
    return np.mean(data), np.var(data)

# --- A3: Minkowski distance plot between 2 samples ---
def minkowski_plot(X):
    v1, v2 = X[0], X[1]
    distances = [minkowski(v1, v2, p) for p in range(1, 11)]
    plt.plot(range(1, 11), distances, marker='o')
    plt.title("A3: Minkowski Distance (r=1 to 10)")
    plt.xlabel("r")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.savefig("A3_minkowski.png")
    plt.close()
    return distances

# --- A4: Split data ---
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)

# --- A5: Train kNN ---
def train_knn(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

# --- A6: Accuracy on test set ---
def evaluate_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)

# --- A7: Predict test set labels ---
def predict_and_display(knn, X_test):
    return knn.predict(X_test)

# --- A8: Vary k from 1 to 11 and plot accuracy ---
def accuracy_vs_k(X_train, y_train, X_test, y_test):
    ks = list(range(1, 12))
    accs = []
    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accs.append(acc)
    plt.plot(ks, accs, marker='o')
    plt.title("A8: Accuracy vs. k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("A8_accuracy_vs_k.png")
    plt.close()
    return ks, accs

# --- A9: Confusion Matrix & Classification Report ---
def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("A9: Confusion Matrix")
    plt.savefig("A9_confusion_matrix.png")
    plt.close()
    report = classification_report(y_test, y_pred, output_dict=False)
    return cm, report

# --- Main Execution ---
if __name__ == "__main__":
    X, y = load_and_preprocess("MC.csv")

    # A1
    c0, c1, s0, s1, dist = analyze_classes(X, y)
    print("\nA1: Intraclass and Interclass Analysis")
    print("Centroid (Class 0):", c0[:5])
    print("Centroid (Class 1):", c1[:5])
    print("Std Dev (Class 0):", s0[:5])
    print("Std Dev (Class 1):", s1[:5])
    print("Interclass Distance:", dist)

    # A2
    mean, var = feature_histogram(X, feature_index=0)
    print("\nA2: Histogram Analysis")
    print("Mean:", mean)
    print("Variance:", var)

    # A3
    distances = minkowski_plot(X)
    print("\nA3: Minkowski distances from r=1 to 10:")
    print(distances)

    # A4
    X_train, X_test, y_train, y_test = split_data(X, y)

    # A5
    knn = train_knn(X_train, y_train, k=3)

    # A6
    acc = evaluate_accuracy(knn, X_test, y_test)
    print("\nA6: Accuracy on test set (k=3): {:.2f}%".format(acc * 100))

    # A7
    y_pred = predict_and_display(knn, X_test)
    print("\nA7: Predictions on Test Set (first 10):", y_pred[:10])

    # A8
    ks, accs = accuracy_vs_k(X_train, y_train, X_test, y_test)
    print("\nA8: Accuracies for k=1 to 11:")
    print(dict(zip(ks, accs)))

    # A9
    cm, report = evaluate_model(y_test, y_pred)
    print("\nA9: Confusion Matrix")
    print(cm)
    print("\nA9: Classification Report")
    print(report)
