import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from MLP import MLP

title = "Нейросеть"

standart_values = {
    "Количество эпох": 50,
    "Скорость обучения": 0.01,
    "Функция активации": "sigmoid",
    "Скрытый слой": 128
}

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title(title)

        self.canvas_width, self.canvas_height = 280, 280
        self.bg_color, self.pen_color = "white", "black"

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color)
        self.canvas.pack(padx=5, pady=5)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_image = ImageDraw.Draw(self.image)

        params_frame = tk.Frame(root)
        params_frame.pack(pady=10)
        params = [("Количество эпох:", standart_values["Количество эпох"]), ("Скорость обучения:", standart_values["Скорость обучения"]), ("Функция активации:", standart_values["Функция активации"]), ( "Скрытый слой:", standart_values["Скрытый слой"])]
        self.entries = {}
        for i, (label, default) in enumerate(params):
            tk.Label(params_frame, text=label).grid(row=i, column=0, padx=5, pady=2, sticky="e")
            entry = tk.Entry(params_frame)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries[label] = entry

        buttons_frame = tk.Frame(root)
        buttons_frame.pack(pady=10)
        buttons = [
            ("Загрузить тренировочный датасет", self.load_train_dataset),
            ("Загрузить валидационный датасет", self.load_val_dataset),
            ("Обучить модель", self.train_model),
            ("Распознать", self.recognize),
            ("Очистить холст", self.clear_canvas)
        ]
        for idx, (text, cmd) in enumerate(buttons):
            tk.Button(buttons_frame, text=text, command=cmd).grid(row=0, column=idx, padx=5)

        self.result_label = tk.Label(root, text="Результат: None", font=("Helvetica", 16))
        self.result_label.pack(pady=5)

        self.model = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.X_train = self.y_train = self.X_val = self.y_val = None
        self.classes = None
        self.mean = self.std = None

    def paint(self, event):
        """Рисование на холсте."""
        x, y = event.x, event.y
        radius = 2
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=self.pen_color, outline=self.pen_color)
        self.draw_image.line([x - radius, y - radius, x + radius, y + radius], fill=0, width=8)

    def clear_canvas(self):
        """Очистка холста и изображения."""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_image = ImageDraw.Draw(self.image)
        self.result_label.config(text="Результат: None")

    def load_dataset(self, dataset_type):
        """Загрузка датасета (тренировочного или валидационного) из CSV-файла."""
        filepath = filedialog.askopenfilename(
            title=f"Выберите CSV файл с {dataset_type} датасетом",
            filetypes=(("CSV файлы", "*.csv"), ("Все файлы", "*.*"))
        )
        if not filepath:
            return

        try:
            data = pd.read_csv(filepath)
            if data.shape[1] < 3:
                messagebox.showerror("Ошибка", f"{dataset_type.capitalize()} датасет должен содержать как минимум 3 столбца: bits, class, id")
                return

            bits = data.iloc[:, 0].apply(self.parse_bits).to_list()
            X = np.array(bits, dtype=np.float32)

            if dataset_type == "тренировочный":
                self.encode_labels(data.iloc[:, 1])
                y = self.encode_classes(data.iloc[:, 1])
                self.X_train, self.y_train = X, self.one_hot_encode(y)
                messagebox.showinfo("Успех", f"Тренировочный датасет загружен успешно!\nКлассы: {', '.join(self.classes)}")
            else:
                if self.X_train is None or self.y_train is None:
                    messagebox.showwarning("Предупреждение", "Пожалуйста, загрузите сначала тренировочный датасет.")
                    return
                y = self.encode_classes(data.iloc[:, 1])
                self.X_val, self.y_val = X, self.one_hot_encode(y)
                messagebox.showinfo("Успех", f"Валидационный датасет загружен успешно!\nКлассы: {', '.join(self.classes)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить {dataset_type} датасет:\n{e}")

    def load_train_dataset(self):
        """Загрузка тренировочного датасета."""
        self.load_dataset("тренировочный")

    def load_val_dataset(self):
        """Загрузка валидационного датасета."""
        self.load_dataset("валидационный")

    def parse_bits(self, bits_str):
        """Парсинг строки битов в список чисел."""
        try:
            if isinstance(bits_str, str):
                bits_str = bits_str.strip("[]")
                return [int(bit) for bit in bits_str.split(",")]
            return list(bits_str)
        except Exception:
            return []

    def encode_labels(self, labels):
        """Кодирование текстовых меток в числовые."""
        unique_labels = sorted(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self.classes = unique_labels

    def encode_classes(self, labels):
        """Преобразование списка меток в числовые индексы."""
        return [self.label_to_index[label] for label in labels]

    def one_hot_encode(self, y):
        """Преобразование меток в one-hot формат."""
        return np.eye(len(self.classes))[y]

    def normalize(self, X):
        """Нормализация признаков (стандартизация)."""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return (X - self.mean) / self.std

    def train_model(self):
        """Обучение модели на загруженном тренировочном датасете."""
        if not all(x is not None and x.size > 0 for x in [self.X_train, self.y_train, self.X_val, self.y_val]):
            messagebox.showwarning("Предупреждение", "Пожалуйста, загрузите тренировочный и валидационный датасеты перед обучением.")
            return

        try:
            epochs = int(self.entries["Количество эпох:"].get())
            lr = float(self.entries["Скорость обучения:"].get())
            activation = self.entries["Функция активации:"].get()
            hidden_size = int(self.entries["Скрытый слой:"].get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные параметры обучения")
            return

        X_train_norm = self.normalize(self.X_train)
        X_val_norm = (self.X_val - self.mean) / self.std

        input_size = X_train_norm.shape[1]
        output_size = self.y_train.shape[1]

        self.model = MLP(input_size, hidden_size, output_size, lr, activation)

        messagebox.showinfo("Обучение", "Начало обучения модели. Это может занять некоторое время.")
        self.model.train(X_train_norm, self.y_train, epochs=epochs, print_loss=True)

        y_val_pred = self.model.predict(X_val_norm)
        y_val_true = np.argmax(self.y_val, axis=1)
        accuracy = np.mean(y_val_pred == y_val_true)

        messagebox.showinfo("Обучение завершено", f"Точность на валидационной выборке: {accuracy * 100:.2f}%")
        self.result_label.config(text=f"Точность на валидации: {accuracy * 100:.2f}%")
        self.model.plot_metrics()

    def recognize(self):
        """Распознавание нарисованного изображения."""
        if self.model is None:
            messagebox.showwarning("Модель не обучена", "Пожалуйста, обучите модель перед распознаванием.")
            return

        cropped_image = self.crop_image()
        processed_image = self.preprocess_image(cropped_image)

        image_array = np.array(cropped_image)
        binary_image = (image_array < 128).astype(float).flatten()
        if np.all(binary_image == 0):
            messagebox.showwarning("Пустое изображение", "Пожалуйста, нарисуйте что-нибудь перед распознаванием.")
            return

        prediction = self.model.predict(processed_image)
        predicted_class = self.index_to_label.get(prediction[0], "Неизвестно")
        self.result_label.config(text=f"Распознанный класс: {predicted_class}")

    def preprocess_image(self, image):
        """Предобработка изображения для модели."""
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = (image_array < 128).astype(float).flatten()
        image_norm = (image_array - self.mean) / self.std
        return image_norm.reshape(1, -1)

    def crop_image(self):
        """Обрезка изображения по краям нарисованного объекта с добавлением отступов и изменение размера до 28x28."""
        np_image = np.array(self.image)
        coords = np.argwhere(np_image < 255)
        if coords.size == 0:
            return self.image.resize((28, 28))

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = self.image.crop((x0, y0, x1, y1))

        padding = 10
        padded_size = (cropped.width + 2 * padding, cropped.height + 2 * padding)
        padded_image = Image.new("L", padded_size, 255)
        padded_image.paste(cropped, (padding, padding))

        resized = padded_image.resize((28, 28))
        # resized.show()
        return resized
