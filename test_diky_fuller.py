import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

# Загрузка данных
data = pd.read_excel('processed_for_etna.xlsx', index_col='Дата', parse_dates=True)

# Выбор ряда для прогнозирования
series = data['Цена']


# Вычисление первой разности
first_diff = series.diff().dropna()

# Вычисление второй разности
second_diff = first_diff.diff().dropna()

# Функция для выполнения теста Дики-Фуллера
def test_stationarity(timeseries):
    print('Результаты теста Дики-Фуллера:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Тестовая статистика', 'p-value', '#Lags Used', 'Количество наблюдений'])
    for key, value in dftest[4].items():
        dfoutput[f'Критическое значение ({key})'] = value
    print(dfoutput)

# # Проверка стационарности исходного ряда
# print("Исходный ряд:")
# test_stationarity(series)
#

# Проверка стационарности после первой разности
print("\nПосле первой разности:")
test_stationarity(first_diff)


# # Проверка стационарности после второй разности
# print("\nПосле второй разности:")
# test_stationarity(second_diff)


acf_values = acf(first_diff.dropna(), nlags=40)
pacf_values = pacf(first_diff.dropna(), nlags=40)

# Создание DataFrame для сохранения значений
acf_pacf_df = pd.DataFrame({
    'Lag': range(41),  # Смещение на один из-за начального лага 0
    'ACF': acf_values,
    'PACF': pacf_values
})

# Сохранение в файл CSV
acf_pacf_df.to_csv('acf_pacf_values.csv', index=False)

print("Значения ACF и PACF сохранены в файл 'acf_pacf_values.csv'.")

seasonally_diffed = series.diff(364).dropna()

s_acf_values = acf(seasonally_diffed.dropna(), nlags=40)
s_pacf_values = pacf(seasonally_diffed.dropna(), nlags=40)

# Создание DataFrame для сохранения значений
s_acf_pacf_df = pd.DataFrame({
    'Lag': range(41),  # Смещение на один из-за начального лага 0
    'ACF': s_acf_values,
    'PACF': s_pacf_values
})

# Сохранение в файл CSV
s_acf_pacf_df.to_csv('s_acf_pacf_values.csv', index=False)

print("Значения ACF и PACF сохранены в файл 'acf_pacf_values.csv'.")