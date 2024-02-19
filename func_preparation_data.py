import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

class TimeSeriesDataset(Dataset):
    """Датасет временных рядов"""
    def __init__(self, prices):
        self.prices = prices

    def __len__(self):
        return len(self.prices) - 1  # Уменьшаем на один, чтобы иметь пространство для y

    def __getitem__(self, idx):
        x = torch.tensor(self.prices[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.prices[idx + 1], dtype=torch.float32).unsqueeze(0)
        return x, y


# Пример модели трансформера
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)  # Добавляем линейный слой для преобразования входных данных
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads,
                                          num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          batch_first=True)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.input_fc(src)  # Преобразование входных данных
        output = self.transformer(src, src)
        output = self.fc(output)
        return output


model = TransformerModel(input_dim=1, model_dim=64, num_heads=4, num_layers=2, output_dim=1)

'''принимает фалй возвращает датафрейм'''
def open_file(file_name)->pd.DataFrame:
    df = pd.read_excel(file_name)
    return df

'''сохраняет датафрейм в файл '''
def save_df_to_excel(df, file_name):
    # Убедитесь, что имя файла имеет расширение .xlsx или .xls
    try:
        df.to_excel(file_name, index=False)  # index=False, чтобы не включать индекс датафрейма в файл

    except Exception as e:
        print(f'Ошибка при сохранении файла: {e}')


'''принимет датафрейм и размер окна смещения и возвращает датафрейм со средними значениями окна'''
def apply_rolling_window(df, size_wind):
    df_output = pd.DataFrame(index=df.index)
    n = len(df)

    for column in df.columns:
        result = []
        for t in range(n):
            start = max(0, t - size_wind // 2)
            end = min(n, t + size_wind // 2 + 1)
            window_mean = np.mean(df.iloc[start:end][column])
            result.append(window_mean)
        df_output[column] = result

    return df_output



