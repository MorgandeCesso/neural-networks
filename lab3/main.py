import numpy as np
import os
import tkinter as tk
from tkinter import Canvas, messagebox, Toplevel, Label, OptionMenu, StringVar
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

np.random.seed(365)

def get_object_bounds(image):
    image_array = np.array(image)
    non_empty_columns = np.where(image_array.min(axis=0) < 255)[0]
    non_empty_rows = np.where(image_array.min(axis=1) < 255)[0]
    if non_empty_columns.any() and non_empty_rows.any():
        upper, lower = non_empty_rows[0], non_empty_rows[-1]
        left, right = non_empty_columns[0], non_empty_columns[-1]
        return left, upper, right, lower
    else:
        return None

def center_object(image):
    bounds = get_object_bounds(image)
    if bounds:
        left, upper, right, lower = bounds
        object_width = right - left
        object_height = lower - upper
        horizontal_padding = (image.width - object_width) // 2
        vertical_padding = (image.height - object_height) // 2
        cropped_image = image.crop(bounds)
        centered_image = Image.new("L", (image.width, image.height), "white")
        centered_image.paste(cropped_image, (horizontal_padding, vertical_padding))
        return centered_image
    return image

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert("L")
                img = center_object(img)
                img = img.resize((100, 100))
                images.append(np.asarray(img).flatten() / 255.0)
                label = int(filename[0])
                labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

def update_weights(weights, x, winner_idx, learning_rate, radius):
    for i in range(weights.shape[1]): # Количество столбцов (weights.shape[1]) соответствует количеству нейронов в сети.
        distance = np.linalg.norm(np.array([winner_idx]) - np.array([i])) # Евклидовое растояние
        if distance <= radius: # Тупо сравниваем дистанцию до нейрона
            influence = np.exp(-distance**2 / (2 * (radius**2))) # Функция ГаУСАААААА
            weights[:, i] += learning_rate * influence * (x - weights[:, i]) # Обновляем веса

def train_som(data, weights, initial_learning_rate, initial_radius, convergence_threshold):
    epoch = 0
    learning_rate = initial_learning_rate
    radius = initial_radius
    flag = True
    while flag:
        prev_weights = weights.copy() # копируем веса до обновления 
        for x in data: # Бегит по картинкам 
            distances = np.linalg.norm(weights - x[:, np.newaxis], axis=0) # Тута считаем дистанцию
            winner_idx = np.argmin(distances) # находим индекс победителя
            update_weights(weights, x, winner_idx, learning_rate, radius) # обновляем веса
        
        weight_change = np.linalg.norm(weights - prev_weights) # считаем изменения весов (сумарное)
        print(f"Epoch: {epoch}, Learning rate: {learning_rate:.6f}, Radius: {radius:.6f}, Weight change: {weight_change:.6f}")
        
        if weight_change < convergence_threshold: # Проверка на мин изменения весов
            print(f"Training converged after {epoch} epochs")
            flag = False
        
        learning_rate *= 0.9
        radius *= 0.9
        epoch += 1
    return weights

def predict(x, weights): # Данная функция нужна для режима работы 
    distances = np.linalg.norm(weights - x[:, np.newaxis], axis=0) # Находит дистанцию до всех нейронов 
    return np.argmin(distances) # Находит победителя с наименьшим растоянием 

class TestCanvas(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), "white")
        self.draw_image = ImageDraw.Draw(self.image)
        test_button = tk.Button(self, text="Проверить", command=self.test_image)
        test_button.pack(side=tk.BOTTOM)
        clear_button = tk.Button(self, text="Очистить", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)
        save_weights = tk.Button(self, text="Сохранить веса", command=self.save_weights)
        save_weights.pack(side=tk.BOTTOM)
        research = tk.Button(self, text="Провести эксперименты", command=self.research)
        research.pack(side=tk.BOTTOM)
        assign_neurons = tk.Button(self, text="Назначить нейроны", command=self.assign_neurons)
        assign_neurons.pack(side=tk.BOTTOM)
        graf = tk.Button(self, text="Показать графики", command=self.graf)
        graf.pack(side=tk.BOTTOM)
        
        self.neuron_assignments = {i: i for i in range(10)}

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-7, y-7, x+7, y+7, fill='black')
        self.draw_image.ellipse([x-7, y-7, x+7, y+7], fill='black')

    def test_image(self):
        centered_image = center_object(self.image)
        inverted_image = centered_image.resize((100, 100))
        img_array = np.array(inverted_image) / 255.0
        img_array = img_array.flatten()
        min_index = predict(img_array, weights)
        assigned_class = self.neuron_assignments[min_index]
        messagebox.showinfo("Результат", f"Это похоже на класс {assigned_class}!")
        self.clear_canvas()

    def save_w(self):
        np.save(f"weights-koh{learning_rate}.npy", weights)
        messagebox.showinfo("Результат", f"Веса сохранены!")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw_image = ImageDraw.Draw(self.image)

    def research(self):
        folder_path = "testing"
        re_images, re_labels = load_images(folder_path)
        wrong_count = 0
        for img, label in zip(re_images, re_labels):
            flag = predict(img, weights)
            assigned_class = self.neuron_assignments[flag]
            if assigned_class != label:
                wrong_count += 1
        messagebox.showinfo("Результат", f"Кол-во ошибок: {wrong_count}, процент ошибок: {(wrong_count/len(re_images))*100:.2f}%")

    def graf(self):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 строки и 5 столбцов
        for i, ax in enumerate(axes.flat):
            if i < weights.shape[1]:  # Проверяем, что индекс не выходит за пределы массива
                ax.imshow(weights[:, i].reshape(100, 100), cmap='gray')
                ax.set_title(f"Нейрон {i}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def save_weights(self):
        np.save(f"weights-koh{learning_rate}.npy", weights)

    def assign_neurons(self):
        assign_window = Toplevel(self)
        assign_window.title("Назначить нейроны")
        labels = []
        options = [str(i) for i in range(10)]
        class_vars = []
        
        for i in range(10):
            Label(assign_window, text=f"Нейрон {i}:").grid(row=i, column=0, padx=10, pady=5)
            class_var = StringVar(assign_window)
            class_var.set(self.neuron_assignments[i])
            class_menu = OptionMenu(assign_window, class_var, *options)
            class_menu.grid(row=i, column=1, padx=10, pady=5)
            class_vars.append(class_var)

        def save_assignments():
            for i, class_var in enumerate(class_vars):
                self.neuron_assignments[i] = int(class_var.get())
            messagebox.showinfo("Результат", "Назначения нейронов сохранены!")
            assign_window.destroy()
        
        save_button = tk.Button(assign_window, text="Сохранить", command=save_assignments)
        save_button.grid(row=10, column=0, columnspan=2, padx=10, pady=10)

if __name__ == "__main__":
    folder_path = "dataset"
    weights = np.random.uniform(-0.3, 0.3, (10000, 10))
    learning_rate = 0.1
    initial_radius = 3.0
    convergence_threshold = 0.1
    try:
        weights = np.load(f"weights-koh{learning_rate}.npy")
        print("Weights loaded successfully")
    except FileNotFoundError:
        images, _ = load_images(folder_path)
        train_som(images, weights, learning_rate, initial_radius, convergence_threshold, 1000)
        print("Weights saved successfully after training")
    app = TestCanvas()
    app.title("Тест нейронной сети.")
    app.mainloop()