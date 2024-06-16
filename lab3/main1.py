import numpy as np
import os
import tkinter as tk
from tkinter import Canvas, messagebox, Toplevel, Label, OptionMenu, StringVar
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

np.random.seed(41)

#Класс с самоорганизующейся картой Кохонена (SOM потому что Self-organized map, ес ай эм фром инглэнд :D)
class SOM:
    def __init__(self, input_dim, num_neurons, learning_rate=0.1, radius=2.0, convergence_threshold=0.1, max_epochs=1000):
        # Сохраняем параметры, которые передали в этот объект SOM
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.radius = radius
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        # Инициализируем веса случайными значениями в диапазоне от -0.3 до 0.3
        self.weights = np.random.uniform(-0.3, 0.3, (input_dim, num_neurons))

    def update_weights(self, x, winner_idx):
        # Проходим по каждому нейрону
        for i in range(self.weights.shape[1]):
            # Считаем расстояние от победившего нейрона до текущего
            distance = np.linalg.norm(np.array([winner_idx]) - np.array([i]))
            # Если нейрон находится в радиусе влияния
            if distance <= self.radius:
                # Считаем влияние (функция Гаусса)
                influence = np.exp(-distance**2 / (2 * (self.radius**2)))
                # Обновляем веса текущего нейрона
                self.weights[:, i] += self.learning_rate * influence * (x - self.weights[:, i])

    def train(self, data):
        # Начинаем обучение
        epoch = 0
        flag = True
        indices = np.random.permutation(len(data))
        data_shuffled = data[indices]
        # Пока не сойдется или не превысит max_epochs
        while flag and epoch < self.max_epochs:
            # Копируем текущие веса перед их обновлением
            prev_weights = self.weights.copy()
            # Проходим по каждому образцу данных
            for x in data_shuffled:
                # Считаем расстояния до всех нейронов
                distances = np.linalg.norm(self.weights - x[:, np.newaxis], axis=0)
                # Ищем нейрон-победитель (с минимальным расстоянием)
                winner_idx = np.argmin(distances)
                # Обновляем веса нейронов
                self.update_weights(x, winner_idx)
            
            # Считаем изменение весов
            weight_change = np.linalg.norm(self.weights - prev_weights)
            print(f"Эпоха: {epoch}, скорость обучения: {self.learning_rate:.6f}, радиус: {self.radius:.6f}, изменение весов: {weight_change:.6f}")
            
            # Проверяем, сошлись ли веса (меньше порога сходимости)
            if weight_change < self.convergence_threshold:
                print(f"Training converged after {epoch} epochs")
                flag = False
            
            # Уменьшаем скорость обучения и радиус
            self.learning_rate *= 0.9
            self.radius *= 0.9
            epoch += 1

    def predict(self, x):
        # Считаем расстояния до всех нейронов
        distances = np.linalg.norm(self.weights - x[:, np.newaxis], axis=0)
        # Возвращаем индекс победившего нейрона
        return np.argmin(distances)

    def save_weights(self, filename):
        # Сохраняем веса в файл
        np.save(filename, self.weights)

    def load_weights(self, filename):
        # Загружаем веса из файла
        self.weights = np.load(filename)

#Класс для обработки картинок (центрирование)
class ImageProcessor:
    @staticmethod
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

    @staticmethod
    def center_object(image):
        bounds = ImageProcessor.get_object_bounds(image)
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

    @staticmethod
    def load_images(folder):
        images = []
        labels = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L")
                    img = ImageProcessor.center_object(img)
                    img = img.resize((100, 100))
                    images.append(np.asarray(img).flatten() / 255.0)
                    label = int(filename[0])
                    labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        return np.array(images), np.array(labels)

#Визуал
class TestCanvas(tk.Tk):
    def __init__(self, som, neuron_assignments):
        super().__init__()
        self.som = som
        self.neuron_assignments = neuron_assignments
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

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-7, y-7, x+7, y+7, fill='black')
        self.draw_image.ellipse([x-7, y-7, x+7, y+7], fill='black')

    def test_image(self):
        centered_image = ImageProcessor.center_object(self.image)
        inverted_image = centered_image.resize((100, 100))
        img_array = np.array(inverted_image) / 255.0
        img_array = img_array.flatten()
        min_index = self.som.predict(img_array)
        assigned_class = self.neuron_assignments[min_index]
        messagebox.showinfo("Результат", f"Это похоже на класс {assigned_class}!")
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw_image = ImageDraw.Draw(self.image)

    def research(self):
        folder_path = "testing"
        re_images, re_labels = ImageProcessor.load_images(folder_path)
        wrong_count = 0
        for img, label in zip(re_images, re_labels):
            flag = self.som.predict(img)
            assigned_class = self.neuron_assignments[flag]
            if assigned_class != label:
                wrong_count += 1
        messagebox.showinfo("Результат", f"Кол-во ошибок: {wrong_count}, процент ошибок: {(wrong_count/len(re_images))*100:.2f}%")

    def graf(self):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            if i < self.som.weights.shape[1]:
                ax.imshow(self.som.weights[:, i].reshape(100, 100), cmap='gray')
                ax.set_title(f"Нейрон {i}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def save_weights(self):
        filename = f"weights-koh{self.som.learning_rate}.npy"
        self.som.save_weights(filename)
        messagebox.showinfo("Результат", f"Веса сохранены в {filename}!")

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
    input_dim = 10000
    num_neurons = 10
    som = SOM(input_dim, num_neurons)
    neuron_assignments = {i: i for i in range(num_neurons)}
    
    try:
        som.load_weights(f"weights-koh{som.learning_rate}.npy")
        print("Weights loaded successfully")
    except FileNotFoundError:
        images, _ = ImageProcessor.load_images(folder_path)
        som.train(images)
        som.save_weights(f"weights-koh{som.learning_rate}.npy")
        print("Weights saved successfully after training")
    
    app = TestCanvas(som, neuron_assignments)
    app.title("Тест нейронной сети.")
    app.mainloop()
