import tkinter as tk
from tkinter import Canvas, messagebox
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

class NumberProcessor():
    def __init__(self, folder, learning_rate, num_classes=10, input_size=10000):
        self.folder = folder
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights_path = f"weights-{self.learning_rate}.npy"

        if os.path.exists(self.weights_path):
            self.weights = np.load(self.weights_path)
            print("Веса успешно загружены")
        else:
            self.weights = np.random.uniform(-0.3, 0.3, (self.num_classes, self.input_size))
            print("Веса не найдены, используются случайно сгенерированные веса")

        self.images, self.lables = self.load_images(folder=self.folder)
        self.train(self.images, self.lables)

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
        bounds = NumberProcessor.get_object_bounds(image)
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
                    img = self.center_object(img)
                    img = img.resize((100, 100))
                    images.append(np.asarray(img).flatten() / 255.0)
                    label = int(filename[0])
                    labels.append(label)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")
        return np.array(images), np.array(labels)

    def guess(self, image):
        sum = [np.dot(w, image) for w in self.weights]
        output = [1 if s > 0 else 0 for s in sum]
        if np.array(output).sum() == 1:
            return output, output.index(1)
        else:
            return output, None

    def train(self, images, labels):
        if self.weights is None:
            self.weights = np.zeros((len(set(labels)), images.shape[1]))
        epoch = 0
        flag = True
        while flag:
            indices = np.random.permutation(len(images))
            images_shuffled = images[indices]
            labels_shuffled = labels[indices]
            flag = False
            for img, label in zip(images_shuffled, labels_shuffled):
                predictions, predicted_label = self.guess(img)
                if predicted_label != label:
                    flag = True
                    self.weights[label] += self.learning_rate * img
                    for i in range(len(self.weights)):
                        if predictions[i] == 1 and i != label:
                            self.weights[i] -= self.learning_rate * img
            epoch += 1
            print(f"Эпоха {epoch}")
        print(f"Обучение завершено после {epoch} эпох")

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
        np.save(f"weights-{self.processor.learning_rate}.npy", self.processor.weights)
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
            pred, flag = self.processor.guess(img)
            if flag is None or flag != label:
                wrong_count += 1
        messagebox.showinfo(
            "Результат",
            f"Количество ошибок: {wrong_count}, процент не распознанных образов: {(wrong_count/len(re_labels))*100:.2f}%"
        )


    def graf(self):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(self.processor.weights[i].reshape((100, 100)), cmap="Greys")
            ax.set_title(f"Веса для класса {i}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    folder_path = "dataset"
    processor = NumberProcessor(folder_path, learning_rate=0.2)
    app = TestCanvas(processor)
    app.title("Распознавание чисел")
    app.mainloop()
