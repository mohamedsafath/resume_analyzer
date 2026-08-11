print("Starting accuracy calculation...")

from sklearn.metrics import accuracy_score

# Example data
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 0, 1]

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

print(f"Model accuracy: {accuracy:.4f}")
