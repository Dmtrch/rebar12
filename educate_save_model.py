
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from func_preparation_data import TimeSeriesDataset
from func_preparation_data import model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Подготовка данных
df = pd.read_excel('data/final_sales_data_corrected.xlsx')
df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%y')
prices = df['Цена'].values.astype(np.float32)  # Преобразование 'Цена' в float

#нормализация
prices = df['Цена'].values.reshape(-1, 1)  # Преобразование в 2D массив для скалера
# Нормализация данных
scaler = MinMaxScaler()
normalized_prices = scaler.fit_transform(prices)

un_scaler = scaler.inverse_transform(normalized_prices)



# Создание экземпляров датасета и даталоадера
dataset = TimeSeriesDataset(normalized_prices)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
#model = TransformerModel(input_dim=1, model_dim=64, num_heads=4, num_layers=2, output_dim=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
model.to(device)

# Цикл обучения (упрощенный)
for epoch in range(num_epochs):
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)  # Перемещение данных на устройство
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'models/model.pth')
