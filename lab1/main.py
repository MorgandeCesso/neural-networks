import numpy as np
import os
import tkinter as tk
from tkinter import Canvas, messagebox
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

np.random.seed(422)

label_dict = {'alpha': 0, 'beta': 1, 'gamma': 2, 'delta': 3, 'epsilon': 4, 
                  'dzeta': 5, 'eta': 6, 'teta': 7, 'yota': 8, 'kappa': 9, 
                  'lambda': 10, 'mu': 11, 'nu': 12, 'xi': 13, 'omicron': 14,
                  'fi': 15, 'ro': 16, 'sigma': 17, 'tau': 18, 'ipsilon': 19,
                  'pi': 20, 'hi': 21, 'psi': 22, 'omega': 23}

inverse_label_dict = {v: k for k, v in label_dict.items()}

def get_object_bounds(image):
    # Преобразование изображения PIL в массив NumPy для обработки
    image_array = np.array(image)
    non_empty_columns = np.where(image_array.min(axis=0) < 255)[0]
    non_empty_rows = np.where(image_array.min(axis=1) < 255)[0]
    if non_empty_columns.any() and non_empty_rows.any():
        upper, lower = non_empty_rows[0], non_empty_rows[-1]
        left, right = non_empty_columns[0], non_empty_columns[-1]
        return left, upper, right, lower
    else:
        return None  # Объект не найден


# Функция для центрирования объекта
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
    global label_dict
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert("L")
                img = center_object(img)
                img = img.resize((100, 100))
                images.append(np.asarray(img).flatten() / 255.0)
                label_name = filename.split('-')[0]  # Имя буквы из названия файла
                label = label_dict[label_name]  # Получение соответствующего индекса
                labels.append(label)
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
    return np.array(images), np.array(labels)


def guess(image, weight, x):
    sum = [np.dot(i, image) for i in weight]
    output = [1 if s > x else 0 for s in sum]
    if np.array(output).sum() == 1:
        return output, output.index(1)
    else:
        return output, None


def train(images, labels, weights, learning_rate):
    epoch = 0
    flag = True
    while flag:
        indices = np.random.permutation(len(images))
        images_shuffled = images[indices]
        labels_shuffled = labels[indices]
        flag = False
        for img, label in zip(images_shuffled, labels_shuffled):
            predictions, predicted_label = guess(img, weights, x=0)
            if predicted_label != label:
                flag = True
                weights[label] += learning_rate * img
                for i in range(len(weights)):
                    if predictions[i] == 1 and i != label:
                        weights[i] -= learning_rate * img
        epoch += 1
        print(f"Эпоха {epoch}")
    print(f"Обучена поле {epoch} эпох")


class TestCanvas(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), "white")
        self.picture_draw = ImageDraw.Draw(self.image)
        test_button = tk.Button(self, text="Проверить", command=self.test_image)
        test_button.pack(side=tk.BOTTOM)
        clear_button = tk.Button(self, text="Очистить", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)
        save_weights = tk.Button(self, text="Сохранить веса", command=self.save_w)
        save_weights.pack(side=tk.BOTTOM)
        research = tk.Button(self, text="Провести тест", command=self.research)
        research.pack(side=tk.BOTTOM)
        graf = tk.Button(self, text="Показать графики", command=self.graf)
        graf.pack(side=tk.BOTTOM)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="black")
        self.picture_draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill="black")

    def test_image(self):
        # Подготовка изображения для модели
        # inverted_image = ImageOps.invert(self.image)
        centered_image = center_object(self.image)
        inverted_image = centered_image.resize((100, 100))
        global inverse_label_dict
        img_array = np.array(inverted_image) / 255.0
        img_array = img_array.flatten()

        result, flag = guess(img_array, weights, x=0)
        if flag == None:
            messagebox.showinfo("Результат", f"Это не похоже не на одну из букв!")
        else:
            messagebox.showinfo("Результат", f"Это похоже на букву {inverse_label_dict[flag]}!")
        self.clear_canvas()

    def save_w(self):
        np.save(f"weights-{learning_rate}.npy", weights)
        messagebox.showinfo("Результат", f"Веса сохранены!")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.picture_draw = ImageDraw.Draw(self.image)

    def research(self):
        folder_path = "test"
        re_images, re_labels = load_images(folder_path)
        wrong_count = 0
        for img, label in zip(re_images, re_labels):
            pred, flag = guess(img, weights, 0)
            if flag == None:
                wrong_count += 1
            elif flag != label:
                wrong_count += 1
        messagebox.showinfo(
            "Результат",
            f"Кол-во ошибок:{wrong_count},процент ошибок:{(wrong_count/480)*100}",
        )

    def graf(self):
        global inverse_label_dict
        fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 8))
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(weights[i].reshape((100, 100)), cmap="Greys")
            ax.set_title(f"Веса для класса {inverse_label_dict[i]}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    folder_path = "dataset"
    weights = np.random.uniform(
        -0.3, 0.3, (24, 10000)
    )  # Веса для 24 классов и изображений 100x100
    learning_rate = 0.01
    try:
        weights = np.load(f"weights-{learning_rate}.npy")
        print("Веса успешно загружены")
    except FileNotFoundError:
        images, labels = load_images(folder_path)
        print("Веса не найдены, инициировано обучение")
        train(images, labels, weights, learning_rate)
    app = TestCanvas()
    app.title("Греческий алфавит")
    app.mainloop()
