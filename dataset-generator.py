import pandas as pd
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузите CSV файл в DataFrame
data = pd.read_csv('annotation.csv', header=None)

# Присвоим названия столбцам для удобства
data.columns = ['image_bits', 'class', 'id']

# Шаг 2: Разделите данные на обучающую и тестовую выборки с сохранением пропорций классов
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    stratify=data['class'],
    random_state=42  # Фиксируем random_state для воспроизводимости
)

# Шаг 3: Сохраните полученные выборки в отдельные CSV файлы
train_data.to_csv('train_dataset.csv', index=False, header=False)
test_data.to_csv('test_dataset.csv', index=False, header=False)

# Дополнительно: Проверка распределения классов в выборках
print("Распределение классов в обучающей выборке:")
print(train_data['class'].value_counts())

print("\nРаспределение классов в тестовой выборке:")
print(test_data['class'].value_counts())
