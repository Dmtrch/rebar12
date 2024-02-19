import pandas as pd
from pmdarima import auto_arima

# Загрузка данных
data = pd.read_excel('sales12bymonthmonth.xls', index_col='Дата', parse_dates=True)

# Выбор ряда для прогнозирования
series = data['Цена']
model = auto_arima(series, start_p=1, start_q=1,
                   max_p=3, max_q=3, m=12,
                   start_P=0, seasonal=True,
                   d=None, D=1, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

print(model.summary())

