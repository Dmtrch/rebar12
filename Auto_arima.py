import pandas as pd
from pmdarima import auto_arima

# Загрузка данных
data = pd.read_excel('processed_for_etna.xlsx', index_col='Дата', parse_dates=True)

# Выбор ряда для прогнозирования
series = data['Цена']

# Автоматический подбор параметров модели SARIMA
model = auto_arima(series, start_p=1, start_q=1,
                   test='adf',
                   max_p=3, max_q=3, m=364,
                   start_P=0, seasonal=True,
                   d=None, D=1, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

# Вывод наилучших параметров
print(model.summary())

# Прогнозирование на 30 дней вперед
forecast = model.predict(n_periods=30)

# Вывод прогноза
print(forecast)
