# predict.py
# This script creates a GUI for drawing digits and predicting them using the
# trained neural network model.
# Make sure to install Pillow: pip install Pillow

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import io

class NeuralNetwork:
    """
    A minimal Neural Network class for loading a pre-trained model and making predictions.
    This is a simplified version of the one in the training script.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        # Placeholders for weights and biases, to be loaded from file
        self.w_ih = None
        self.w_ho = None
        self.b_h = None
        self.b_o = None
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def load_model(self, filename='mnist_model.npz'):
        """Loads weights and biases from a .npz file."""
        try:
            data = np.load(filename)
            self.w_ih = data['w_ih']
            self.w_ho = data['w_ho']
            self.b_h = data['b_h']
            self.b_o = data['b_o']
            return True
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file not found: {filename}\nPlease run neural_network_trainer.py first.")
            return False

    def feedforward(self, inputs):
        """Propagates input through the network."""
        hidden_inputs = np.dot(self.w_ih, inputs) + self.b_h
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.w_ho, hidden_outputs) + self.b_o
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def predict(self, inputs):
        """Makes a prediction for a given input."""
        final_outputs = self.feedforward(inputs)
        idx = np.argmax(final_outputs)
        return idx, final_outputs[idx][0]


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        
        # --- UI Elements ---
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit, font=('Arial', 12))
        self.predict_button.grid(row=1, column=0, padx=5, pady=10, sticky="ew")

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas, font=('Arial', 12))
        self.clear_button.grid(row=1, column=1, padx=5, pady=10, sticky="ew")

        self.result_label = tk.Label(root, text="Draw a digit (0-9)", font=('Arial', 14, 'bold'))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

        # --- Drawing Setup ---
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # --- Load Neural Network ---
        self.nn = self.load_network()

    def load_network(self):
        """Loads the pre-trained neural network model."""
        # Load model parameters to instantiate the network correctly
        try:
            params = np.load('mnist_model.npz')
            nn = NeuralNetwork(
                input_nodes=int(params['input_nodes']),
                hidden_nodes=int(params['hidden_nodes']),
                output_nodes=int(params['output_nodes'])
            )
            if nn.load_model():
                return nn
            else:
                self.root.destroy() # Close app if model isn't found
                return None
        except FileNotFoundError:
             messagebox.showerror("Error", "Model file not found: mnist_model.npz\nPlease run neural_network_trainer.py first.")
             self.root.destroy()
             return None


    def paint(self, event):
        """Draws on both the Tkinter canvas and the PIL image."""
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        # Draw on the visible canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        # Draw on the hidden PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill="black", width=10)
        self.predict_digit()

    def clear_canvas(self):
        """Clears the canvas and the PIL image."""
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")
        self.result_label.config(text="Draw a digit (0-9)")

    def predict_digit(self):
        """Processes the drawn image and uses the NN to predict the digit."""
        if not self.nn:
            return

        # --- Preprocess the image for the network ---
        # 1. Invert colors (network was trained on white digits on black background)
        img = ImageOps.invert(self.image)
        
        # 2. Resize to 28x28 pixels, using antialiasing for better quality
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 3. Convert image to numpy array
        img_array = np.array(img)
        
        # 4. Normalize the data to the range [0.01, 1.0]
        # This matches the preprocessing from the training script
        normalized_array = (img_array / 255.0 * 0.99) + 0.01
        
        # 5. Flatten the 28x28 array into a 784x1 vector
        inputs = normalized_array.flatten().reshape(784, 1)

        # --- Make a prediction ---
        prediction, prediction_quality = self.nn.predict(inputs)
        
        # Update the result label
        self.result_label.config(text=f"Prediction: {prediction} ({prediction_quality * 100:0.1f}% sure)")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
