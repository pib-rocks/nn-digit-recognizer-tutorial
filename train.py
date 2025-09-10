# neural_network_trainer.py
# This script trains a neural network on the MNIST dataset using only NumPy.
# It then saves the trained weights and biases to a file.

import numpy as np
import gzip
import os
import urllib.request
import time

class NeuralNetwork:
    """
    A simple Neural Network with one hidden layer.
    
    Attributes:
        input_nodes (int): Number of neurons in the input layer.
        hidden_nodes (int): Number of neurons in the hidden layer.
        output_nodes (int): Number of neurons in the output layer.
        learning_rate (float): The step size for gradient descent.
        w_ih (np.array): Weights matrix for input-to-hidden layer.
        w_ho (np.array): Weights matrix for hidden-to-output layer.
        b_h (np.array): Bias vector for the hidden layer.
        b_o (np.array): Bias vector for the output layer.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """Initializes the network with random weights and biases."""
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Initialize weights with random values, scaled to the number of nodes
        # This helps prevent vanishing/exploding gradients initially.
        self.w_ih = np.random.randn(self.hidden_nodes, self.input_nodes) * np.sqrt(1. / self.input_nodes)
        self.w_ho = np.random.randn(self.output_nodes, self.hidden_nodes) * np.sqrt(1. / self.hidden_nodes)

        # Initialize biases with zeros
        self.b_h = np.zeros((self.hidden_nodes, 1))
        self.b_o = np.zeros((self.output_nodes, 1))

        # Activation function (sigmoid)
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def feedforward(self, inputs):
        """Propagates input through the network."""
        # --- Input to Hidden Layer ---
        # Calculate the weighted sum of inputs and add bias
        hidden_inputs = np.dot(self.w_ih, inputs) + self.b_h
        # Apply the activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        # --- Hidden to Output Layer ---
        # Calculate the weighted sum of hidden layer outputs and add bias
        final_inputs = np.dot(self.w_ho, hidden_outputs) + self.b_o
        # Apply the activation function to get the final network output
        final_outputs = self.activation_function(final_inputs)

        return hidden_outputs, final_outputs

    def train(self, inputs, targets):
        """
        Trains the network using backpropagation.
        
        Args:
            inputs (np.array): The input data (e.g., a flattened image).
            targets (np.array): The target output (e.g., a one-hot encoded label).
        """
        # Perform a feedforward pass to get the outputs
        hidden_outputs, final_outputs = self.feedforward(inputs)

        # --- Calculate Errors ---
        # Output layer error is (target - actual)
        output_errors = targets - final_outputs
        # Hidden layer error is the output_errors, split by weights,
        # propagated back to the hidden layer.
        hidden_errors = np.dot(self.w_ho.T, output_errors)

        # --- Backpropagate and Update Weights & Biases ---
        # This is the core of backpropagation (gradient descent)
        
        # Calculate gradients for hidden-to-output weights
        # Gradient = learning_rate * error * (output * (1 - output)) * hidden_outputs.T
        # The (output * (1-output)) part is the derivative of the sigmoid function
        gradient_ho = self.learning_rate * output_errors * final_outputs * (1 - final_outputs)
        
        # Update hidden-to-output weights and biases
        self.w_ho += np.dot(gradient_ho, hidden_outputs.T)
        self.b_o += gradient_ho

        # Calculate gradients for input-to-hidden weights
        gradient_ih = self.learning_rate * hidden_errors * hidden_outputs * (1 - hidden_outputs)
        
        # Update input-to-hidden weights and biases
        self.w_ih += np.dot(gradient_ih, inputs.T)
        self.b_h += gradient_ih

    def predict(self, inputs):
        """Makes a prediction for a given input."""
        _, final_outputs = self.feedforward(inputs)
        # The prediction is the index of the neuron with the highest activation
        return np.argmax(final_outputs)


def download_mnist(path="mnist_data"):
    """Downloads and extracts the MNIST dataset if not already present."""
    base_url = "https://github.com/pib-rocks/nn-digit-recognizer-tutorial/raw/refs/heads/main/data/"
    files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

    for file in files:
        file_path = os.path.join(path, file)
        if not os.path.exists(file_path.replace('.gz', '')):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, file_path)
            print(f"Extracting {file}...")
            with gzip.open(file_path, 'rb') as f_in:
                with open(file_path.replace('.gz', ''), 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(file_path)
    print("MNIST dataset is ready.")

def load_mnist(path="mnist_data"):
    """Loads the MNIST data from the specified path."""
    def load_images(filename):
        with open(filename, 'rb') as f:
            # Read metadata
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
        return images

    def load_labels(filename):
        with open(filename, 'rb') as f:
            # Read metadata
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    train_images = load_images(os.path.join(path, "train-images-idx3-ubyte"))
    train_labels = load_labels(os.path.join(path, "train-labels-idx1-ubyte"))
    test_images = load_images(os.path.join(path, "t10k-images-idx3-ubyte"))
    test_labels = load_labels(os.path.join(path, "t10k-labels-idx1-ubyte"))

    return (train_images, train_labels), (test_images, test_labels)


def preprocess_data(images, labels):
    """Normalizes images and one-hot encodes labels."""
    # Normalize pixel values from [0, 255] to [0.01, 1.0]
    # We use 0.01 instead of 0 to avoid zero inputs which can kill gradients
    processed_images = (images / 255.0 * 0.99) + 0.01
    # Flatten the 28x28 images into a 784x1 vector
    processed_images = processed_images.reshape(len(images), -1)

    # Create one-hot encoded labels
    processed_labels = np.zeros((len(labels), 10))
    for i, label in enumerate(labels):
        processed_labels[i, label] = 0.99 # Use 0.99 instead of 1
    
    return processed_images, processed_labels


if __name__ == "__main__":
    # --- Network & Training Parameters ---
    INPUT_NODES = 784  # 28 * 28 pixels
    HIDDEN_NODES = 200
    OUTPUT_NODES = 10 # Digits 0-9
    LEARNING_RATE = 0.1
    EPOCHS = 5
    
    # --- 1. Load and Prepare Data ---
    print("Checking for MNIST dataset...")
    download_mnist()
    (train_images, train_labels), (test_images, test_labels) = load_mnist()

    print("Preprocessing data...")
    X_train, y_train = preprocess_data(train_images, train_labels)
    X_test, y_test_raw = preprocess_data(test_images, test_labels) # Keep raw labels for eval

    # --- 2. Initialize the Neural Network ---
    print("Initializing the neural network...")
    nn = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

    # --- 3. Train the Network ---
    print(f"Starting training for {EPOCHS} epochs...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        print(f"  Epoch {epoch + 1}/{EPOCHS}")
        for i in range(len(X_train)):
            # Reshape input and target to be column vectors
            inputs = X_train[i].reshape(INPUT_NODES, 1)
            targets = y_train[i].reshape(OUTPUT_NODES, 1)
            nn.train(inputs, targets)
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # --- 4. Evaluate the Network ---
    print("Evaluating network accuracy...")
    correct_predictions = 0
    for i in range(len(X_test)):
        inputs = X_test[i].reshape(INPUT_NODES, 1)
        prediction = nn.predict(inputs)
        if prediction == test_labels[i]:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(X_test)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    # --- 5. Save the Trained Model ---
    model_filename = 'mnist_model.npz'
    print(f"Saving trained model to {model_filename}...")
    np.savez(
        model_filename, 
        w_ih=nn.w_ih, 
        w_ho=nn.w_ho, 
        b_h=nn.b_h, 
        b_o=nn.b_o,
        input_nodes=INPUT_NODES,
        hidden_nodes=HIDDEN_NODES,
        output_nodes=OUTPUT_NODES,
        learning_rate=LEARNING_RATE # Not used for prediction but good for reference
    )
    print("Model saved successfully. You can now run digit_recognizer_app.py.")
