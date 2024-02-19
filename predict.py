import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from func_preparation_data import model


device = 'cpu'
def make_forecast(model, input_data, forecast_depth, device='cpu'):
    predictions = []
    with torch.no_grad():  # Отключение градиентов
        # Преобразование numpy массива в тензор и добавление необходимых измерений
        current_input = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Удаление .unsqueeze(-1)
        current_input = current_input.to(device)
        for _ in range(forecast_depth):
            prediction = model(current_input)
            # Добавляем предсказание к списку
            predictions.append(prediction.cpu().numpy())  # Перемещаем предсказания на CPU и преобразуем в numpy
            # Здесь можно добавить логику обновления current_input для последовательного прогнозирования

    return predictions


# Загрузка состояния модели
model.load_state_dict(torch.load('models/model.pth'))

# Переключение модели в режим оценки
model.eval()

# Подготовка данных
df = pd.read_excel('data/final_sales_data_corrected.xlsx')
df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%y')
prices = df['Цена'].values.astype(np.float32)  # Преобразование 'Цена' в float

#нормализация
prices = df['Цена'].values.reshape(-1, 1)  # Преобразование в 2D массив для скалера
# Нормализация данных
scaler = MinMaxScaler()
normalized_prices = scaler.fit_transform(prices)

forecast = make_forecast(model, normalized_prices[2195:], 1, device)

forecast_absolute = []
for pred in forecast:
    # Преобразование каждого прогноза отдельно и добавление его в список
    pred_absolute = scaler.inverse_transform(pred.reshape(-1, 1))
    forecast_absolute.append(pred_absolute)

# Теперь forecast_absolute содержит список ненормализованных прогнозов
print(forecast_absolute)

