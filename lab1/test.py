from tkinter import Canvas, Scrollbar, messagebox
from typing import List
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

alpha = 0.2


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
                if output != check_goal(training_imgs[i][1], neuron_i.symbol):
                    neuron_i.update_weights(
                        training_imgs[i][0],
                        check_goal(training_imgs[i][1], neuron_i.symbol),
                        output,
                    )
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
    img = 255 - img  # type: ignore
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
        training_images.append((inputs, filename_str.split("-")[0]))
        exist = False
        for n in all_neurons:
            if n.symbol == filename_str.split("-")[0]:
                exist = True
        if not exist:
            np.random.seed(5)
            all_neurons.append(
                Neuron(
                    [np.random.uniform(-0.3, 0.3) for _ in range(10000)],
                    filename_str.split("-")[0],
                )
            )
    np.random.seed(5)
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
        visualize_button = tk.Button(
            self, text="Visualize Weights", command=self.visualize
        )
        visualize_button.pack(side=tk.BOTTOM)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="black")
        self.image_draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill="black")

    def test_image(self):
        resized_image = self.image.resize((100, 100))
        resized_image.save("pixel_art.bmp")
        img_array = image_to_array("pixel_art.bmp")
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
        neuron_height = 4  # Высота каждого изображения нейрона в дюймах
        neuron_width = 4  # Ширина каждого изображения нейрона в дюймах
        fig, axes = plt.subplots(
            num_neurons, 1, figsize=(neuron_width, num_neurons * neuron_height)
        )

        if num_neurons == 1:
            axes = [axes]  # Преобразование в список, если нейрон один

        for ax, neuron in zip(axes, all_neurons):
            weight_matrix = np.array(neuron.weights).reshape(100, 100)
            ax.imshow(weight_matrix, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Symbol: {neuron.symbol}")

        # Создание нового окна с канвой для скроллинга
        new_window = tk.Toplevel(self)
        new_window.title("Weights Visualization")

        canvas = Canvas(
            new_window, width=300, height=600
        )  # Устанавливаем ширину и высоту канвы
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Добавление вертикального скроллбара
        v_scrollbar = Scrollbar(new_window, orient=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=v_scrollbar.set)

        # Добавление фигуры на канву
        fig_canvas_agg = FigureCanvasTkAgg(fig, canvas)
        widget = fig_canvas_agg.get_tk_widget()
        widget.pack()

        # Конфигурация канвы и скроллбара
        canvas.create_window((0, 0), window=widget, anchor="nw")
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Обеспечиваем возможность скроллинга мышью
        canvas.bind_all(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"),
        )


if __name__ == "__main__":
    # Start the training when the application launches
    start_training()
    app = TestCanvas()
    app.title("Neural Network Test: Greek Letters")
    app.mainloop()
