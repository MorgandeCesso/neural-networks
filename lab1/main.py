import numpy as np
import os
import tkinter as tk
from tkinter import Canvas, messagebox
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

# Инициализация весов для 24 нейронов (букв греческого алфавита) и изображений размером 100x100
weights = np.random.uniform(-0.3, 0.3, (24, 10000))
learning_rate = 0.2  # Скорость обучения

# Функция для загрузки изображений из папки
def load_images(folder):
    images = []
    labels = []
    # Словарь для сопоставления названия буквы к её индексу
    label_dict = {'alpha': 0, 'beta': 1, 'gamma': 2, 'delta': 3, 'epsilon': 4, 
                  'dzeta': 5, 'eta': 6, 'teta': 7, 'yota': 8, 'kappa': 9, 
                  'lambda': 10, 'mu': 11, 'nu': 12, 'xi': 13, 'omicron': 14,
                  'fi': 15, 'ro': 16, 'sigma': 17, 'tau': 18, 'ipsilon': 19,
                  'pi': 20, 'hi': 21, 'psi': 22, 'omega': 23}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert("L")  # Конвертация в градации серого
                img = img.resize((100, 100))  # Изменение размера до 100x100
                images.append(np.asarray(img).flatten() / 255.0)  # Нормализация и преобразование в вектор
                label_name = filename.split('-')[0]  # Имя буквы из названия файла
                label = label_dict[label_name]  # Получение соответствующего индекса
                labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

# Функция для вычисления предсказания на основе текущих весов
def guess(image, weight):
    sum = [np.dot(i, image) for i in weight]  # Скалярное произведение весов и входного изображения
    output = [1 if s >= 0 else 0 for s in sum]  # Пороговая функция активации
    if np.array(output).sum() > 1:
        return output, output.index(1)
    else:
        return output, None

# Функция обучения нейронной сети
def train(images, labels, weights, learning_rate):
    epoch = 0
    while True:
        indices = np.random.permutation(len(images))
        images_shuffled = images[indices]
        labels_shuffled = labels[indices]
        wrong_predictions = 0
        for img, label in zip(images_shuffled, labels_shuffled):
            predictions, predicted_label = guess(img, weights)
            if predicted_label == label:
                wrong_predictions += 1
                if predictions == img and predicted_label != None:
                    weights[label] += learning_rate * img * (1 - predictions[int(predicted_label)])  # Увеличение весовых коэфов правильного нейрона
                elif predictions != img and predicted_label != None:
                    weights[label] += learning_rate * img * (0 - predictions[int(predicted_label)])  # Увеличение весовых коэфов правильного нейрона
        if wrong_predictions == 0:
            print(f"Perfectly trained after {epoch} epochs")
            break
        epoch += 1
        print(f"Epoch {epoch}: Number of wrong predictions = {wrong_predictions}")

def visualize_weights(weights):
    num_classes, size = weights.shape
    side = int(np.sqrt(size))  # Предполагаем, что изображения квадратные

    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8))  # Настраиваем под 24 нейрона
    fig.suptitle('Визуализация весов для каждого нейрона', fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        if i < num_classes:
            weight_image = weights[i].reshape(side, side)
            im = ax.imshow(weight_image, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Нейрон {i}')
            ax.axis('off')
        else:
            ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.01)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

# GUI для взаимодействия с пользователем
class TestCanvas(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), "white")
        self.image_draw = ImageDraw.Draw(self.image)
        test_button = tk.Button(self, text="Проверить", command=self.test_image)
        test_button.pack(side=tk.BOTTOM)
        clear_button = tk.Button(self, text="Очистить", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)
        save_weights_button = tk.Button(self, text="Сохранить веса", command=self.save_weights)
        save_weights_button.pack(side=tk.BOTTOM)
        # research_button = tk.Button(self, text="Провести эксперименты", command=self.research)
        # research_button.pack(side=tk.BOTTOM)
        visualize_button = tk.Button(self, text="Визуализировать веса", command=self.visualize)
        visualize_button.pack(side=tk.BOTTOM)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='black')
        self.image_draw.ellipse([x-5, y-5, x+5, y+5], fill='black')

    def test_image(self):
        resized_image = self.image.resize((100, 100))
        img_array = np.array(resized_image) / 255.0
        img_array = img_array.flatten()
        result, flag = guess(img_array, weights)
        if flag is None:
            messagebox.showinfo("Результат", "Это не похоже не на одну из букв!")
        else:
            messagebox.showinfo("Результат", f"Это похоже на букву {flag}!")
        self.clear_canvas()

    def save_weights(self):
        np.save(f"weights-{learning_rate}.npy", weights)
        messagebox.showinfo("Результат", "Веса сохранены!")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.image_draw = ImageDraw.Draw(self.image)
    
    def visualize(self):
        visualize_weights(weights)

    # def research(self):
    #     folder_path = "TestSamples2"
    #     re_images, re_labels = load_images(folder_path)
    #     wrong_count = 0
    #     for img, label in zip(re_images, re_labels):
    #         pred, flag = guess(img, weights)
    #         if flag is None or flag != label:
    #             wrong_count += 1
    #     total_images = len(re_images)
    #     error_rate = (wrong_count / total_images) * 100
    #     messagebox.showinfo("Результат эксперимента", f"Кол-во ошибок: {wrong_count}, процент ошибок: {error_rate:.2f}%")

if __name__ == "__main__":
    # Путь к папке с изображениями для обучения
    training_folder = "dataset"
    images, labels = load_images(training_folder)
    train(images, labels, weights, learning_rate)  # Обучение нейронной сети

    app = TestCanvas()
    app.title("Тест нейронной сети: Греческие буквы")
    app.mainloop()
