import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Загрузка данных

# Загрузка данных
data = pd.read_excel('processed_for_etna.xlsx', index_col='Дата', parse_dates=True)

# Выбор ряда для прогнозирования
series = data['Цена']

# Определение диапазонов параметров для перебора
p_range = q_range = range(0, 3)  # Для примера рассматриваем p и q от 0 до 2
P_range = Q_range = range(0, 2)  # Для P и Q рассматриваем от 0 до 1
D = 1  # Предполагаем, что D=1 на основании предыдущего анализа

# Инициализация переменных для записи лучших результатов
best_aic = float('inf')
best_bic = float('inf')
best_cfg = None

# Перебор комбинаций параметров
for p in p_range:
    for q in q_range:
        for P in P_range:
            for Q in Q_range:
                try:
                    # Обучение модели SARIMA с текущей комбинацией параметров
                    model = SARIMAX(series, order=(p, D, q), seasonal_order=(P, D, Q, 364))
                    model_fit = model.fit(disp=False)

                    # Сохранение модели, если ее AIC меньше предыдущего лучшего
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_bic = model_fit.bic
                        best_cfg = (p, q, P, Q)
                        print(f'Новая лучшая модель: SARIMA{best_cfg} AIC={best_aic:.3f}, BIC={best_bic:.3f}')
                except:
                    continue

print(f'Лучшая модель: SARIMA{best_cfg} с AIC={best_aic:.3f} и BIC={best_bic:.3f}')
