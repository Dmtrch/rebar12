import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Загрузка данных
data = pd.read_excel('sales12to24.xls', index_col='Дата', parse_dates=True)
y = data['Цена']

# Обучение модели SARIMA
model = SARIMAX(y, order=(2,2,0), seasonal_order=(2,2,1,12))
results = model.fit()

# Прогнозирование
forecast = results.forecast(steps=4)
print(forecast)