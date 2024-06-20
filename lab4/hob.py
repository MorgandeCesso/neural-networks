import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt


# Сама нейронка Хопфилда
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size  # Размер
        self.weights = np.zeros((size, size))  # Веса

    def train(self, patterns):
        for pattern in patterns: 
            self.weights += np.outer(pattern, pattern)  # Загрузка образов
        np.fill_diagonal(
            self.weights, 0
        )  

    def update(self, state):
        return np.sign(np.dot(self.weights, state))  # Обновляем состояние сети через пороговую функцию (тета ноль)

    def run(self, state):
        previous_state = np.zeros_like(state)
        flag = True 
        while flag:
            new_state = self.update(state)  
            if np.array_equal(
                new_state, previous_state
            ): 
                flag = False  
            previous_state = state  
            state = new_state 
        return state

    def unlearn(self, pattern):
        self.weights -= np.outer(
            pattern, pattern
        )
        np.fill_diagonal(
            self.weights, 0
        )


# Зашумление
def add_noise(image, noise_level=0.1):
    noisy_image = image.copy()
    num_noisy_pixels = int(noise_level * image.size)
    noisy_indices = np.random.choice(image.size, num_noisy_pixels, replace=False)
    noisy_image[noisy_indices] *= -1
    return noisy_image


# Недофронт
class HopfieldApp(QWidget):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.patterns = []
        self.current_image = None
        self.current_image_index = -1
        self.noise_level = 0.1
        self.initUI()
        self.load_images()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel(self)
        layout.addWidget(self.label)

        hbox = QHBoxLayout()
        self.image_buttons = []
        for i in range(3):
            btn = QPushButton(f"Изображение {i + 1}", self)
            btn.clicked.connect(lambda _, x=i: self.select_image(x))
            hbox.addWidget(btn)
            self.image_buttons.append(btn)
        layout.addLayout(hbox)

        self.selected_image_label = QLabel("Выбрано изображение: Нет", self)
        self.selected_image_label.setFont(QFont("Arial", 24))
        self.selected_image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self.selected_image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.selected_image_label)

        self.train_button = QPushButton("Обучить", self)
        self.train_button.clicked.connect(self.train)
        layout.addWidget(self.train_button)

        self.run_button = QPushButton("Запустить", self)
        self.run_button.clicked.connect(self.run)
        layout.addWidget(self.run_button)

        self.noise_button = QPushButton("Добавить шум", self)
        self.noise_button.clicked.connect(self.add_noise)
        layout.addWidget(self.noise_button)

        self.unlearn_button = QPushButton("Разобучить", self)
        self.unlearn_button.clicked.connect(self.unlearn)
        layout.addWidget(self.unlearn_button)

        self.setLayout(layout)
        self.setWindowTitle("Сеть Хопфилда")
        self.show()

    def load_images(self):
        folder = "dataset"
        patterns = []
        for filename in os.listdir(folder):
            if filename.endswith(".bmp"):
                filepath = os.path.join(folder, filename)
                image = QImage(filepath)
                if image.width() == 20 and image.height() == 20:
                    pattern = self.image_to_pattern(image)
                    patterns.append(pattern)
        self.patterns = patterns
        print("Изображения загружены")

    def select_image(self, index):
        if index < len(self.patterns):
            self.current_image = self.patterns[index]
            self.current_image_index = index
            self.noise_level = 0.1
            self.selected_image_label.setText(f"Выбрано изображение: {index + 1}")
            self.display_image(self.current_image)

    def image_to_pattern(self, image):
        gray_image = image.convertToFormat(QImage.Format_Grayscale8)
        buffer = gray_image.bits()
        buffer.setsize(gray_image.byteCount())
        array = np.frombuffer(buffer, dtype=np.uint8).reshape((20, 20))
        pattern = np.where(array > 127, 1, -1).flatten()
        return pattern

    def train(self):
        self.network.train(self.patterns)
        print("Обучение завершено")

    def run(self):
        if self.current_image is not None:
            result = self.network.run(self.current_image)
            self.display_image(result)

    def add_noise(self):
        if self.current_image is not None:
            self.noise_level += 0.1
            noisy_image = add_noise(self.current_image, self.noise_level)
            self.display_image(noisy_image)

    def unlearn(self):
        if self.current_image is not None:
            self.network.unlearn(self.current_image)
            print("Разобучение завершено")

    def display_image(self, image):
        img = (image.reshape(20, 20) * 255).astype(np.uint8)
        qimage = QImage(img, 20, 20, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(100, 100, Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    network = HopfieldNetwork(400)
    ex = HopfieldApp(network)
    sys.exit(app.exec_())
