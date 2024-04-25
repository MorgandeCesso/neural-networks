from tkinter import messagebox
from typing import List
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

alpha = 0.4


def f_x_step(x):
    if x > 0:
        return 1
    return 0


def check_goal(current_symbol, neuron_symbol):
    if current_symbol == neuron_symbol:
        return 1
    return 0


def start_training():
    training_imgs = make_vectors()
    for neuron_i in all_neurons:
        print(neuron_i.symbol)
        error_found = True
        epoch = -1
        while error_found:
            error_found = False
            print("Epoch #", epoch)
            for i in range(len(training_imgs)):
                output = neuron_i.forward(training_imgs[i][0])
                # print(f"[{neuron_i.weights[25]}][{neuron_i.weights[30]}][{neuron_i.weights[35]}][{neuron_i.weights[40]}][{neuron_i.weights[45]}][{neuron_i.weights[50]}], i[{i}], neuron: {neuron_i.symbol}, img_sym: {training_imgs[i][1]}, out: {output}, goal: {check_goal(training_imgs[i][1], neuron_i.symbol)}")
                if output != check_goal(training_imgs[i][1], neuron_i.symbol):
                    neuron_i.update_weights(training_imgs[i][0], check_goal(training_imgs[i][1], neuron_i.symbol),
                                            output)
                    error_found = True
            epoch += 1


class Neuron:
    def __init__(self, inp_weights, r_symbol):
        self.weights = inp_weights
        self.alpha = alpha
        self.symbol = r_symbol

    def forward(self, inputs):
        weighted_sum = 0
        for i in range(len(self.weights)):
            weighted_sum += self.weights[i] * inputs[i]
        return f_x_step(weighted_sum)

    def update_weights(self, inputs, goal, output):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.alpha * (goal - output) * inputs[i]


np.random.seed(5)
all_neurons: List[Neuron] = []


def image_to_array(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(img)
    img = 255 - img # type: ignore
    img //= 255
    img_flat = img.flatten()
    img_array = np.array(img_flat, dtype=np.uint8)
    return img_array


def make_vectors():
    train_dir = "dataset"
    image_filenames = os.listdir(train_dir)
    training_images = []
    for filename in image_filenames:
        filename_str = str(filename)  # Преобразование объекта в строку
        image_path = os.path.join(train_dir, filename_str)
        inputs = image_to_array(image_path)
        training_images.append((inputs, filename_str.split('-')[0]))
        exist = False
        for n in all_neurons:
            if n.symbol == filename_str.split('-')[0]:
                exist = True
        if not exist:
            np.random.seed(5)
            all_neurons.append(Neuron([np.random.uniform(-0.3, 0.3) for _ in range(10000)], filename_str.split('-')[0]))
    np.random.shuffle(training_images)
    return training_images


class TestCanvas(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), "white")
        self.image_draw = ImageDraw.Draw(self.image)

        test_button = tk.Button(self, text="Test", command=self.test_image)
        test_button.pack(side=tk.BOTTOM)
        clear_button = tk.Button(self, text="Clear", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)
        visualize_button = tk.Button(self, text="Visualize Weights", command=self.visualize)
        visualize_button.pack(side=tk.BOTTOM)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill='black')
        self.image_draw.ellipse([x-8, y-8, x+8, y+8], fill='black')

    def test_image(self):
        resized_image = self.image.resize((100, 100))
        resized_image.save('pixel_art.bmp')
        img_array = image_to_array('pixel_art.bmp')
        guessed_symbol = None
        max_response = 0
        for neuron in all_neurons:
            response = neuron.forward(img_array)
            print(img_array)
            if response > max_response:
                max_response = response
                guessed_symbol = neuron.symbol
        if guessed_symbol is not None:
            messagebox.showinfo("Result", f"It looks like the letter {guessed_symbol}!")
        else:
            messagebox.showinfo("Result", "This does not resemble any known symbol!")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.image_draw = ImageDraw.Draw(self.image)

    def visualize(self):
        num_neurons = len(all_neurons)
        fig, axes = plt.subplots(1, num_neurons, figsize=(num_neurons * 2, 2))

        if num_neurons == 1:
            axes = [axes]  # Make it iterable if only one neuron.

        for ax, neuron in zip(axes, all_neurons):
            # Assuming each neuron has 10000 weights arranged as a 100x100 image
            weight_matrix = np.array(neuron.weights).reshape(100, 100)
            ax.imshow(weight_matrix, cmap='gray')
            ax.axis('off')  # Turn off axis
            ax.set_title(f'Symbol: {neuron.symbol}')

        plt.show()


if __name__ == "__main__":
    # Start the training when the application launches
    start_training()
    app = TestCanvas()
    app.title("Neural Network Test: Greek Letters")
    app.mainloop()
