import pandas as pd
from datetime import datetime, timedelta

# Чтение данных из файла Excel
file_path = 'sales12bymonthclear.xls'
sales_data_df = pd.read_excel(file_path)

# Преобразование столбца с датами в строковый формат 'dd.mm.yy'
sales_data_df['дата'] = pd.to_datetime(sales_data_df.iloc[:, 0]).dt.strftime('%d.%m.%y')
# Создание словаря с датами и значениями продаж и цен
sales_data = dict(zip(sales_data_df['дата'], zip(sales_data_df.iloc[:, 1], sales_data_df.iloc[:, 2])))

# Создание диапазона дат
start_date = datetime(2018, 1, 1)
end_date = datetime(2024, 2, 13)
date_range = pd.date_range(start=start_date, end=end_date).strftime('%d.%m.%y')

# Подготовка новых данных для файла
new_sales_data = []
last_known_price = 0
for current_date in date_range:
    if current_date in sales_data:
        quantity, price = sales_data[current_date]
        last_known_price = price
    else:
        quantity = 0.001
        price = last_known_price
    new_sales_data.append([current_date, quantity, price])

# Создание итогового DataFrame и сохранение его в файл Excel
final_sales_df = pd.DataFrame(new_sales_data, columns=['Дата', 'Количество', 'Цена'])
final_sales_df.to_excel('final_sales_data_corrected.xlsx', index=False)


