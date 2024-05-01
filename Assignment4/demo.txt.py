import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv("lab04_dataset_3.csv")

# Step 2: Split the dataset into training and testing sets
X = data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y = data['quality_grp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define function to train MLPClassifier with different architectures
def train_mlp(activation, hidden_layers, neurons, X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=tuple(neurons for _ in range(hidden_layers)),
                        activation=activation,
                        max_iter=10000,
                        random_state=2023484)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    misclassification_rate = 1 - accuracy_score(y_test, y_pred)
    return misclassification_rate

# Step 4: Iterate over different configurations and compute misclassification rate
results = []
activations = ['logistic', 'relu', 'tanh']
hidden_layers = [1, 2, 3, 4, 5]
neurons_per_layer = [2, 4, 6, 8]

for activation in activations:
    for hidden_layer in hidden_layers:
        for neurons in neurons_per_layer:
            print(activation, ' ', hidden_layer, ' ', neurons)
            misclassification_rate = train_mlp(activation, hidden_layer, neurons, X_train, y_train, X_test, y_test)
            results.append([activation, hidden_layer, neurons, misclassification_rate])

# Step 5: Output the results in a dataframe
results_df = pd.DataFrame(results, columns=['Activation Function', 'Hidden layers', 'Neurons per layer', 'Misclassification Rate'])
print(results_df)
