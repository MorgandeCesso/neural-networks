import tkinter as tk
from tkinter import Canvas, messagebox
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

class NumberProcessor():
    def __init__(self, folder, initial_learning_rate=0.01, hidden_size=256, num_classes=10, input_size=10000, regularization_lambda=0.01, decay_rate=0.999, dropout_prob=0.5):
        self.folder = folder  # Папка с изображениями
        self.initial_learning_rate = initial_learning_rate  # Начальная скорость обучения
        self.learning_rate = initial_learning_rate  # Текущая скорость обучения
        self.input_size = input_size  # Размер входного слоя (100x100 изображение -> 10000 пикселей)
        self.hidden_size = hidden_size  # Размер скрытого слоя
        self.num_classes = num_classes  # Количество классов (цифр 0-9)
        self.regularization_lambda = regularization_lambda  # Коэффициент регуляризации
        self.decay_rate = decay_rate  # Коэффициент затухания скорости обучения
        self.dropout_prob = dropout_prob  # Вероятность Dropout
        self.weights_path = f"weights-{self.initial_learning_rate}.npz"  # Путь к файлу для сохранения весов

        if os.path.exists(self.weights_path):
            # Если файл весов существует, загружаем его
            data = np.load(self.weights_path)
            self.W1 = data['W1']  # Веса для первого слоя
            self.b1 = data['b1']  # Смещения для первого слоя
            self.W2 = data['W2']  # Веса для второго слоя
            self.b2 = data['b2']  # Смещения для второго слоя
            print("Веса успешно загружены")
        else:
            # Инициализация весов и смещений случайными значениями
            self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)  # He инициализация
            self.b1 = np.zeros((1, self.hidden_size))  # Нулевые смещения для первого слоя
            self.W2 = np.random.randn(self.hidden_size, self.num_classes) * np.sqrt(2. / self.hidden_size)  # He инициализация
            self.b2 = np.zeros((1, self.num_classes))  # Нулевые смещения для второго слоя
            print("Веса не найдены, используются случайно сгенерированные веса")

        # Загрузка изображений и меток
        self.images, self.labels = self.load_images(folder=self.folder)
        # Визуализация первых 5 изображений
        self.visualize_images(self.images[:5], self.labels[:5])
        # Обучение модели
        self.train(self.images, self.labels)

    @staticmethod
    def get_object_bounds(image):
        # Получение границ объекта на изображении
        image_array = np.array(image)  # Преобразование изображения в массив
        non_empty_columns = np.where(image_array.min(axis=0) < 255)[0]  # Ненулевые столбцы
        non_empty_rows = np.where(image_array.min(axis=1) < 255)[0]  # Ненулевые строки
        if non_empty_columns.any() and non_empty_rows.any():
            upper, lower = non_empty_rows[0], non_empty_rows[-1]  # Верхняя и нижняя границы
            left, right = non_empty_columns[0], non_empty_columns[-1]  # Левая и правая границы
            return left, upper, right, lower
        else:
            return None

    @staticmethod
    def center_object(image):
        # Центрирование объекта на изображении
        bounds = NumberProcessor.get_object_bounds(image)
        if bounds:
            left, upper, right, lower = bounds
            object_width = right - left  # Ширина объекта
            object_height = lower - upper  # Высота объекта
            horizontal_padding = (image.width - object_width) // 2  # Горизонтальные отступы
            vertical_padding = (image.height - object_height) // 2  # Вертикальные отступы
            cropped_image = image.crop(bounds)  # Обрезка изображения по границам объекта
            centered_image = Image.new("L", (image.width, image.height), "white")  # Новое изображение с белым фоном
            centered_image.paste(cropped_image, (horizontal_padding, vertical_padding))  # Вставка обрезанного изображения на центр
            return centered_image
        return image

    def load_images(self, folder=None):
        # Загрузка изображений из папки
        if folder is None:
            folder = self.folder
        images = []
        labels = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L")  # Преобразование изображения в оттенки серого
                    img = self.center_object(img)  # Центрирование объекта на изображении
                    img = img.resize((100, 100))  # Изменение размера изображения на 100x100
                    images.append(np.asarray(img).flatten() / 255.0)  # Преобразование изображения в массив и нормализация
                    label = int(filename[0])  # Предполагаем, что метка закодирована в имени файла
                    labels.append(label)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")
        return np.array(images), np.array(labels)

    def visualize_images(self, images, labels):
        # Визуализация изображений
        fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
        for img, lbl, ax in zip(images, labels, axes):
            ax.imshow(img.reshape((100, 100)), cmap="gray")
            ax.set_title(f"Label: {lbl}")
            ax.axis("off")
        plt.show()

    def forward(self, X, training=True):
        # Прямое распространение
        Z1 = np.dot(X, self.W1) + self.b1  # Линейная комбинация входных данных и весов первого слоя
        A1 = np.maximum(0, Z1)  # ReLU активация

        # Применение Dropout
        if training:
            D1 = (np.random.rand(*A1.shape) < self.dropout_prob) / self.dropout_prob  # Создание маски Dropout
            A1 *= D1  # Применение Dropout
        else:
            D1 = np.ones_like(A1)  # Без Dropout на этапе тестирования

        Z2 = np.dot(A1, self.W2) + self.b2  # Линейная комбинация скрытого слоя и весов второго слоя
        A2 = self.softmax(Z2)  # Softmax активация
        return Z1, A1, D1, Z2, A2

    def softmax(self, x):
        # Softmax функция активации
        e_x = np.exp(x - np.max(x))  # Экспонента каждого элемента с вычитанием максимального для численной стабильности
        return e_x / e_x.sum(axis=1, keepdims=True)  # Нормализация для получения вероятностей

    def backward(self, X, Y, Z1, A1, D1, Z2, A2):
        # Обратное распространение
        m = Y.shape[0]  # Количество образцов

        dZ2 = A2 - Y  # Градиент потерь по Z2
        dW2 = np.dot(A1.T, dZ2) / m  # Градиент потерь по W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Градиент потерь по b2

        dA1 = np.dot(dZ2, self.W2.T)  # Градиент потерь по A1
        dZ1 = dA1 * (Z1 > 0)  # Градиент потерь по Z1 с учетом производной ReLU
        dZ1 *= D1  # Применение Dropout маски
        dW1 = np.dot(X.T, dZ1) / m  # Градиент потерь по W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Градиент потерь по b1

        # Добавление L2 регуляризации
        dW2 += self.regularization_lambda * self.W2
        dW1 += self.regularization_lambda * self.W1

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        # Обновление параметров
        self.W1 -= self.learning_rate * dW1  # Обновление весов первого слоя
        self.b1 -= self.learning_rate * db1  # Обновление смещений первого слоя
        self.W2 -= self.learning_rate * dW2  # Обновление весов второго слоя
        self.b2 -= self.learning_rate * db2  # Обновление смещений второго слоя

    def guess(self, image):
        # Предсказание метки для изображения
        _, _, _, _, A2 = self.forward(image, training=False)  # Прямое распространение
        return A2, np.argmax(A2, axis=1)[0]  # Возвращаем вероятности и предсказанную метку

    def train(self, images, labels):
        # Обучение модели
        Y = np.eye(self.num_classes)[labels]  # Преобразование меток в one-hot вектор
        epoch = 0
        max_epochs = 5000  # Максимальное количество эпох
        while epoch < max_epochs:
            self.learning_rate = self.initial_learning_rate * (self.decay_rate ** epoch)  # Экспоненциальное затухание скорости обучения
            Z1, A1, D1, Z2, A2 = self.forward(images)  # Прямое распространение
            loss = self.cross_entropy_loss(Y, A2)  # Вычисление потерь
            dW1, db1, dW2, db2 = self.backward(images, Y, Z1, A1, D1, Z2, A2)  # Обратное распространение
            self.update_parameters(dW1, db1, dW2, db2)  # Обновление параметров
            
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, потери: {loss}, скорость обучения: {self.learning_rate}")  # Вывод потерь и скорости обучения каждые 100 эпох
            
            if loss < 0.01:  # Условие остановки по малому значению потерь (можно настроить в зависимости от данных)
                break
            epoch += 1
        print(f"Обучение завершено после {epoch} эпох")

    def cross_entropy_loss(self, y_true, y_pred):
        # Вычисление потерь с использованием кросс-энтропии
        epsilon = 1e-10  # Малое значение для избежания деления на ноль
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Ограничение значений y_pred
        n_samples = y_true.shape[0]  # Количество образцов
        log_p = -np.log(y_pred[np.arange(n_samples), y_true.argmax(axis=1)])  # Логарифм предсказанных вероятностей для правильных меток
        loss = np.sum(log_p) / n_samples  # Среднее значение потерь по всем образцам
        return loss

    def save_weights(self):
        # Сохранение весов в файл
        np.savez(self.weights_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def visualize_weights(self):
        # Визуализация весов
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(self.W1[:, i].reshape((100, 100)), cmap="Greys")
            ax.set_title(f"Веса для класса {i}")
        plt.tight_layout()
        plt.show()

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
        graf = tk.Button(self, text="Визуализация", command=self.graf)
        graf.pack(side=tk.BOTTOM)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="black")
        self.picture_draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill="black")

    def test_image(self):
        centered_image = self.processor.center_object(self.image)
        resized_image = centered_image.resize((100, 100))
        img_array = np.array(resized_image) / 255.0
        img_array = img_array.flatten()

        result, flag = self.processor.guess(img_array)
        if flag == None:
            messagebox.showinfo("Результат", "Страшно, очень страшно, мы не знаем, что это такое!")
        else:
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

    def graf(self):
        self.processor.visualize_weights()

if __name__ == "__main__":
    folder_path = "dataset"
    processor = NumberProcessor(folder_path, initial_learning_rate=0.1)  # Начальная скорость обучения
    app = TestCanvas(processor)
    app.title("Распознавание чисел")
    app.mainloop()
