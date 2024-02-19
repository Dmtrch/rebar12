import pandas as pd

# Загрузка данных из файла
file_path = 'final_sales_data_corrected.xlsx'
data = pd.read_excel(file_path)

# Преобразование столбца 'Дата' в datetime и установка его как индекса DataFrame
data['Дата'] = pd.to_datetime(data['Дата'], format='%d.%m.%y')
data.set_index('Дата', inplace=True)

# Убедимся, что данные в столбцах 'Количество' и 'Цена' имеют правильный тип float
data['Количество'] = data['Количество'].astype(float)
data['Цена'] = data['Цена'].astype(float)

# Сохранение обработанных данных в новый файл
processed_file_path = 'processed_for_etna.xlsx'
data.to_excel(processed_file_path)

print(f"Данные успешно обработаны и сохранены в файл: {processed_file_path}")
