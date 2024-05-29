import tkinter as tk
from tkinter import Canvas, messagebox
from PIL import Image, ImageDraw
import numpy as np
import os

class NumberProcessor():
    def __init__(self, folder):
        self.folder = folder  # Папка с изображениями
        self.learning_rate = 0.01  # Скорость обучения
        self.input_size = 10000  # Размер входного слоя (100x100 изображение -> 10000 пикселей)
        self.hidden_size = 128  # Размер скрытого слоя
        self.num_classes = 10  # Количество классов (цифр 0-9)

        # Инициализация весов и смещений случайными значениями
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.num_classes) * 0.01
        self.b2 = np.zeros((1, self.num_classes))

        # Загрузка изображений и меток
        self.images, self.labels = self.load_images(folder=self.folder)
        # Обучение модели
        self.train(self.images, self.labels)

    def load_images(self, folder=None):
        if folder is None:
            folder = self.folder
        images = []
        labels = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L")
                    img = img.resize((100, 100))
                    images.append(np.asarray(img).flatten() / 255.0)
                    label = int(filename[0])
                    labels.append(label)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")
        return np.array(images), np.array(labels)

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = Y.shape[0]

        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def guess(self, image):
        _, _, _, A2 = self.forward(image)
        return A2, np.argmax(A2, axis=1)[0]

    def train(self, images, labels):
        Y = np.eye(self.num_classes)[labels]
        epoch = 0
        max_epochs = 500
        while epoch < max_epochs:
            Z1, A1, Z2, A2 = self.forward(images)
            loss = self.cross_entropy_loss(Y, A2)
            dW1, db1, dW2, db2 = self.backward(images, Y, Z1, A1, Z2, A2)
            self.update_parameters(dW1, db1, dW2, db2)
            
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, потери: {loss}")
            
            if loss < 0.1:
                break
            epoch += 1
        print(f"Обучение завершено после {epoch} эпох")

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        n_samples = y_true.shape[0]
        log_p = -np.log(y_pred[np.arange(n_samples), y_true.argmax(axis=1)])
        loss = np.sum(log_p) / n_samples
        return loss

    def save_weights(self):
        np.savez("weights.npz", W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

class TestCanvas(tk.Tk):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.canvas = Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), "white")
        self.picture_draw = ImageDraw.Draw(self.image)
        test_button = tk.Button(self, text="Распознание числа", command=self.test_image)
        test_button.pack(side=tk.BOTTOM)
        clear_button = tk.Button(self, text="Очистить", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)
        save_weights = tk.Button(self, text="Сохранить весовые коэффициенты", command=self.save_w)
        save_weights.pack(side=tk.BOTTOM)
        research = tk.Button(self, text="Тестирование", command=self.research)
        research.pack(side=tk.BOTTOM)
    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="black")
        self.picture_draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill="black")

    def test_image(self):
        resized_image = self.image.resize((100, 100))
        img_array = np.array(resized_image) / 255.0
        img_array = img_array.flatten()

        result, flag = self.processor.guess(img_array)
        messagebox.showinfo("Результат", f"Это похоже на цифру {flag}!")
        self.clear_canvas()

    def save_w(self):
        self.processor.save_weights()
        messagebox.showinfo("Результат", "Весовые коэффициенты сохранены!")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.picture_draw = ImageDraw.Draw(self.image)

    def research(self):
        test_folder = "testing"
        re_images, re_labels = self.processor.load_images(test_folder)
        wrong_count = 0
        for img, label in zip(re_images, re_labels):
            _, flag = self.processor.guess(img)
            if flag is None or flag != label:
                wrong_count += 1
        messagebox.showinfo(
            "Результат",
            f"Количество ошибок: {wrong_count}, процент не распознанных образов: {(wrong_count/len(re_labels))*100:.2f}%"
        )

if __name__ == "__main__":
    folder_path = "dataset"
    processor = NumberProcessor(folder_path)
    app = TestCanvas(processor)
    app.title("Распознавание чисел")
    app.mainloop()