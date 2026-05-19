# Import the required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load a built-in sample dataset (Iris flower dataset)
print("Loading data...")
dataset = load_iris()
X = dataset.data  # Features (flower measurements)
y = dataset.target  # Labels (flower species)

# 2. Split the data into Training data (80%) and Testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize a Machine Learning Model (Random Forest)
print("Training the model...")
model = RandomForestClassifier()

# 4. Train the model using the training data
model.fit(X_train, y_train)

# 5. Make predictions on the test data
predictions = model.predict(X_test)

# 6. Check how accurate the model is
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")
