import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import Canvas, messagebox
from PIL import Image, ImageDraw, ImageOps
from typing import Tuple, List

class PerceptronModel:
    def __init__(self, input_size: int, learning_rate: float = 0.5) -> None:
        self.weights: np.ndarray = np.random.rand(input_size) # Весовые коэффициенты
        self.learning_rate: float = learning_rate # Альфа

    def threshold_activation(self, x: float) -> int:
        return 1 if x >= 0 else 0 # Пороговая функция

    def predict(self, input_data: np.ndarray) -> int:
        weighted_sum: float = np.dot(input_data, self.weights) # Считаем взвешенную сумму
        return self.threshold_activation(weighted_sum) # Применяем активационную функцию

    def train(self, input_data: np.ndarray, target: np.ndarray) -> None:
        for i in range(len(input_data)):
            weighted_sum: float = np.dot(input_data[i], self.weights) # Взвешенная сумма для текущего образца
            activated_output: int = self.threshold_activation(weighted_sum) # Выходные данные активируются 
            error: int = target[i] - activated_output # Считаем ошибку
            if error != 0: 
                self.weights += input_data[i] * error * self.learning_rate # Корректируем веса, если ошибка не = 0

    def train_until_perfect(self, images: np.ndarray, labels: np.ndarray, max_epochs: int = 100) -> None:
        for epoch in range(max_epochs): # Во избежание зацикливания лимит эпох 100
            indices: np.ndarray = np.arange(len(images))
            np.random.shuffle(indices) # Шаффл индексов до рандомного порядка
            images_shuffled: np.ndarray = images[indices] 
            labels_shuffled: np.ndarray = labels[indices]
            wrong_count: int = sum(self.predict(images_shuffled[i]) != labels_shuffled[i] for i in range(len(images_shuffled))) # Считаем число ошибок
            epoch_info: str = f"Epoch {epoch+1}, wrong predictions: {wrong_count}"
            if wrong_count == 0:
                print(epoch_info)
                print(f"Perfectly trained after {epoch+1} epochs")
                return # Выходим, если ошибок нет
            self.train(images_shuffled, labels_shuffled) # Обучаем
            print(epoch_info)
        print("Stopped training after reaching the maximum number of epochs.")

def load_images(folder: str) -> Tuple[np.ndarray, np.ndarray]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    for filename in os.listdir(folder):
        img_path: str = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img: Image.Image = img.convert("L") # Конвертация в черно-белое изображение
                img: Image.Image = img.resize((100, 100)) # Изменение размера изображения до 100x100
                images.append(np.asarray(img).flatten() / 255.0) # Преобразование изображения в нормализованный одномерный массив
                label: int = 0 if int(filename.split('.')[0]) < 20 else 1 # Присвоение метки в зависимости от имени файла
                labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

# Рисуем крестики с ноликами
class TestCanvas(tk.Tk):
    def __init__(self, model: PerceptronModel) -> None:
        super().__init__()
        self.model: PerceptronModel = model
        self.canvas: Canvas = Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.bind("<B1-Motion>", self.draw_on_canvas)
        self.image: Image.Image = Image.new("L", (280, 280), "white")
        self.image_draw: ImageDraw.ImageDraw = ImageDraw.Draw(self.image)
        self.setup_buttons()

    def setup_buttons(self) -> None:
        test_button: tk.Button = tk.Button(self, text="Проверить", command=self.test_image)
        test_button.pack(side=tk.BOTTOM)
        clear_button: tk.Button = tk.Button(self, text="Очистить", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)

    def draw_on_canvas(self, event: tk.Event) -> None:
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='black')
        self.image_draw.ellipse([x-10, y-10, x+10, y+10], fill='black')
    
    def test_image(self) -> None:
        inverted_image: Image.Image = self.image
        inverted_image: Image.Image = inverted_image.resize((100, 100))
        img_array: np.ndarray = np.array(inverted_image) / 255.0
        img_array = img_array.flatten()
        result: int = self.model.predict(img_array)
        if result == 0:
            messagebox.showinfo("Результат", "Это похоже на крестик!")
        else:
            messagebox.showinfo("Результат", "Это похоже на нолик!")
        self.clear_canvas()

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.image_draw = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    folder_path: str = "dataset"
    images, labels = load_images(folder_path)
    model: PerceptronModel = PerceptronModel(10000)  # 100x100 пикселей
    model.train_until_perfect(images, labels)
    app: TestCanvas = TestCanvas(model)
    app.title("Тест нейронной сети: Нолик или Крестик")
    app.mainloop()
    matrix: np.ndarray = model.weights.reshape(100, 100)
    print(matrix)
    plt.imshow(matrix, cmap='gray')  # Визуализация весов
    plt.colorbar()
    plt.title("Визуализация весовых коэффициентов")
    plt.show()
