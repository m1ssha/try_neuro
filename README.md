# Запуск

Скопировать репо
```
git clone https://github.com/m1ssha/try_neuro.git
```

Поставить виртуальное окружение
```
python -m venv venv
venv/Scripts/activate
```

Установить зависимости

```
pip install requirements.txt
```

Запустить
```
python main.py
```

Жрёт датасеты в формате annotation.csv (в репозитории).
Датасет делится на две выборки (тренировочный и валидационный) и загружается два разных csv файла