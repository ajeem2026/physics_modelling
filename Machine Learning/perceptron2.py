import numpy as np

# ------------------------------------------------------------------------------
# Once upon a time in a magical land, there were two kinds of creatures:
# Dragons and Unicorns. These creatures could be identified by two magical traits:
#
# 1. Fire Power: How powerful their fiery breath is.
# 2. Sparkle Level: How much they sparkle in the sunlight.
#
# In our magical kingdom:
# - Dragons are known for their high fire power but low sparkle.
# - Unicorns, on the other hand, are known for their low fire power and high sparkle.
#
# Our goal is to build a perceptron—a simple model—that learns to distinguish 
# between dragons and unicorns based on these two features.
# ------------------------------------------------------------------------------

def initialize_weights(n_features, random_state=None):
    """
    Initialize the weights for the perceptron, including a bias term.
    
    Parameters:
        n_features (int): Number of features for each sample (in our case, 2: fire_power and sparkle_level).
        random_state (int, optional): Seed for reproducibility.
    
    Returns:
        weights (ndarray): A weight vector of size n_features+1 (first element is the bias).
    """
    rgen = np.random.RandomState(random_state)
    # Initialize weights with small random numbers
    weights = rgen.normal(loc=0.0, scale=0.01, size=n_features + 1)
    return weights

def net_input(x, weights):
    """
    Compute the net input (weighted sum) for a single sample.
    
    Parameters:
        x (ndarray): 1D array of features.
        weights (ndarray): Weight vector where weights[0] is the bias.
    
    Returns:
        float: The net input value.
    """
    return np.dot(x, weights[1:]) + weights[0]

def predict(x, weights):
    """
    Make a prediction for a single sample.
    
    Parameters:
        x (ndarray): 1D array of features.
        weights (ndarray): Weight vector.
    
    Returns:
        int: Predicted label: 1 for Dragon, -1 for Unicorn.
    """
    return 1 if net_input(x, weights) >= 0.0 else -1

def perceptron_train(X, y, learning_rate=0.1, n_epochs=10, random_state=None):
    """
    Train the perceptron model on the provided dataset.
    
    Parameters:
        X (ndarray): 2D array of features (each row is a sample).
        y (ndarray): 1D array of target labels (1 for Dragon, -1 for Unicorn).
        learning_rate (float): Step size for weight updates.
        n_epochs (int): Number of passes through the dataset.
        random_state (int, optional): Seed for reproducibility.
    
    Returns:
        weights (ndarray): The learned weight vector.
    """
    n_features = X.shape[1]
    weights = initialize_weights(n_features, random_state)
    
    # Loop over the dataset for a specified number of epochs
    for epoch in range(n_epochs):
        for xi, target in zip(X, y):
            # Predict the label using current weights
            prediction = predict(xi, weights)
            # Compute the error (difference between actual and predicted label)
            error = target - prediction
            # Update the feature weights using the perceptron learning rule
            weights[1:] += learning_rate * error * xi
            # Update the bias (first weight)
            weights[0] += learning_rate * error
        # Print weights after each epoch to observe the learning process
        print(f"Epoch {epoch+1}/{n_epochs}, Weights: {weights}")
    return weights

if __name__ == '__main__':
    # ------------------------------------------------------------------------------
    # Our Fun Magical Creatures Dataset:
    # We have collected data on magical creatures, and each creature is described by
    # two features: [fire_power, sparkle_level]
    #
    # For Dragons (labeled as 1):
    #   - They have high fire power and low sparkle.
    #   Examples: [8, 2], [7, 1], [9, 3]
    #
    # For Unicorns (labeled as -1):
    #   - They have low fire power and high sparkle.
    #   Examples: [1, 8], [2, 9], [3, 7]
    # ------------------------------------------------------------------------------
    X = np.array([
        [8, 2],  # Dragon: high fire power, low sparkle
        [7, 1],  # Dragon
        [9, 3],  # Dragon
        [1, 8],  # Unicorn: low fire power, high sparkle
        [2, 9],  # Unicorn
        [3, 7]   # Unicorn
    ])
    # Define the target labels: 1 for Dragon, -1 for Unicorn
    y = np.array([1, 1, 1, -1, -1, -1])
    
    # Train the perceptron model on our magical creatures dataset
    weights = perceptron_train(X, y, learning_rate=0.1, n_epochs=10, random_state=42)
    
    # ------------------------------------------------------------------------------
    # Testing the Trained Model:
    # Let's introduce a new magical creature with features:
    #   [fire_power = 6, sparkle_level = 4]
    #
    # This creature has moderate fire power and sparkle.
    # The trained perceptron will decide whether this creature is a Dragon or a Unicorn.
    # ------------------------------------------------------------------------------
    new_creature = np.array([6, 4])
    prediction = predict(new_creature, weights)
    
    # Interpret the prediction based on our labeling convention
    creature_type = "Dragon" if prediction == 1 else "Unicorn"
    print("\nThe new creature with features [fire_power=6, sparkle_level=4] is classified as:", creature_type)
