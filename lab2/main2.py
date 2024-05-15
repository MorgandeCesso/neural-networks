import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
class Neuron:
    def __init__(self, num_inputs, index, neuron_type):
        self.weights = np.random.uniform(-0.3, 0.3, num_inputs)
        self.output = 0.0
        self.error = 0.0
        self.index = index
        self.type = neuron_type

    def save_weights(self):
        with open(f'Weights[{self.type}-{self.index}].txt', 'w') as f:
            for weight in self.weights:
                f.write(f'{weight}\n')

    def load_weights(self):
        if os.path.exists(f'Weights[{self.type}-{self.index}].txt'):
            with open(f'Weights[{self.type}-{self.index}].txt', 'r') as f:
                self.weights = np.array([float(line.strip()) for line in f])
        else:
            self.weights = np.random.uniform(-0.3, 0.3, len(self.weights))
            self.save_weights()

    def activate(self, inputs):
        s = np.dot(self.weights, inputs)
        self.output = 1 / (1 + np.exp(-0.5 * s))

    def calculate_error_output(self, expected):
        self.error = self.output - expected

    def calculate_error_hidden(self, next_error, next_output, next_weight):
        self.error += next_error * next_output * (1 - next_output) * next_weight

    def update_weights(self, alpha, prev_output):
        self.weights -= alpha * self.error * self.output * (1 - self.output) * np.array(prev_output)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, hidden_size2=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.neurons_hidden = [Neuron(input_size, i, 'hidden') for i in range(hidden_size)]
        self.neurons_hidden2 = [Neuron(hidden_size, i, 'hidden2') for i in range(hidden_size2)] if hidden_size2 else []
        self.neurons_output = [Neuron(hidden_size2 if hidden_size2 else hidden_size, i, 'output') for i in range(output_size)]

        for neuron in self.neurons_hidden:
            neuron.load_weights()
        for neuron in self.neurons_hidden2:
            neuron.load_weights()
        for neuron in self.neurons_output:
            neuron.load_weights()

    def feed_forward(self, inputs):
        hidden_outputs = np.array([neuron.activate(inputs) or neuron.output for neuron in self.neurons_hidden])
        if self.hidden_size2:
            hidden_outputs2 = np.array([neuron.activate(hidden_outputs) or neuron.output for neuron in self.neurons_hidden2])
            final_outputs = np.array([neuron.activate(hidden_outputs2) or neuron.output for neuron in self.neurons_output])
        else:
            final_outputs = np.array([neuron.activate(hidden_outputs) or neuron.output for neuron in self.neurons_output])
        return final_outputs

    def back_propagate(self, inputs, expected_outputs, alpha):
        hidden_outputs = np.array([neuron.output for neuron in self.neurons_hidden])
        if self.hidden_size2:
            hidden_outputs2 = np.array([neuron.output for neuron in self.neurons_hidden2])
            for i, neuron in enumerate(self.neurons_output):
                neuron.calculate_error_output(expected_outputs[i])
                neuron.update_weights(alpha, hidden_outputs2)

            for i, neuron in enumerate(self.neurons_hidden2):
                neuron.error = 0.0
                for next_neuron in self.neurons_output:
                    neuron.calculate_error_hidden(next_neuron.error, next_neuron.output, next_neuron.weights[i])
                neuron.update_weights(alpha, hidden_outputs)

            for i, neuron in enumerate(self.neurons_hidden):
                neuron.error = 0.0
                for next_neuron in self.neurons_hidden2:
                    neuron.calculate_error_hidden(next_neuron.error, next_neuron.output, next_neuron.weights[i])
                neuron.update_weights(alpha, inputs)
        else:
            for i, neuron in enumerate(self.neurons_output):
                neuron.calculate_error_output(expected_outputs[i])
                neuron.update_weights(alpha, hidden_outputs)

            for i, neuron in enumerate(self.neurons_hidden):
                neuron.error = 0.0
                for next_neuron in self.neurons_output:
                    neuron.calculate_error_hidden(next_neuron.error, next_neuron.output, next_neuron.weights[i])
                neuron.update_weights(alpha, inputs)

    def train(self, training_data, epochs, epsilon, alpha):
        for epoch in range(epochs):
            total_error = 0.0
            for inputs, targets in training_data:
                self.feed_forward(inputs)
                self.back_propagate(inputs, targets, alpha)
                total_error += sum((neuron.error ** 2) / 2 for neuron in self.neurons_output)
            total_error /= len(training_data)
            print(f"Эпоха: {epoch}, ошибка: {total_error}")
            if total_error < epsilon:
                print(f"Обучение завершено на эпохе {epoch}")        
                break
        for neuron in self.neurons_output:
            neuron.save_weights()
        for neuron in self.neurons_hidden:
            neuron.save_weights()
        for neuron in self.neurons_hidden2:
            neuron.save_weights()

class Application(tk.Tk):
    def __init__(self, neural_network):
        super().__init__()
        self.title("Neural Network Digit Recognition")
        self.geometry("300x400")
        self.nn = neural_network
        self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
        self.canvas.pack()
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(fill=tk.X)
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)
        self.recognize_button = tk.Button(self.button_frame, text="Recognize", command=self.recognize)
        self.recognize_button.pack(side=tk.LEFT)
        self.train_button = tk.Button(self.button_frame, text="Train", command=self.train)
        self.train_button.pack(side=tk.LEFT)
        self.result_label = tk.Label(self, text="", font=("Helvetica", 24))
        self.result_label.pack()

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def preprocess_image(self):
        img = self.image.resize((28, 28)).convert('L')
        img = np.array(img).reshape(784) / 255.0
        return img

    def recognize(self):
        inputs = self.preprocess_image()
        outputs = self.nn.feed_forward(inputs)
        recognized_digit = np.argmax(outputs)
        self.result_label.config(text=str(recognized_digit))

    def train(self):
        training_data = []
        for label in range(10):
            for i in range(30):  # 30 изображений для обучения на каждую цифру
                img = Image.open(f'dataset/{label}-{i}.bmp').convert('L')
                img = img.resize((28, 28))
                img_data = np.array(img).reshape(784) / 255.0
                target = [0] * 10
                target[label] = 1
                training_data.append((img_data, target))

        self.nn.train(training_data, epochs=100, epsilon=0.005, alpha=0.1)
        self.result_label.config(text="Training completed")

if __name__ == "__main__":
    input_size = 784  # 28x28 pixels
    hidden_size = 512
    hidden_size2 = 256
    output_size = 10  # Digits 0-9
    nn = NeuralNetwork(input_size, hidden_size, output_size, hidden_size2)
    app = Application(nn)
    app.mainloop()