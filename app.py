import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Загружаем обученную модель
model = joblib.load('./model/gradient_boosting_model.pkl')

# Загружаем scaler
scaler = joblib.load('./model/scaler.pkl')

# Загружаем датасет
df = pd.read_csv('data/predictive_maintenance.csv')

# Получаем минимальные и максимальные значения температуры
min_air_temp = df['Air temperature'].min()
max_air_temp = df['Air temperature'].max()

# Заголовок приложения
st.title('Прогнозирование значения Torque')
st.write('Введите значения признаков для прогнозирования')

# Ввод признаков через Streamlit
air_temperature = st.slider('Температура воздуха', min_value=int(min_air_temp), max_value=int(max_air_temp))
process_temperature = st.slider('Температура процесса', min_value=int(df['Process temperature'].min()), max_value=int(df['Process temperature'].max()))
rotational_speed = st.slider('Скорость вращения', min_value=int(df['Rotational speed'].min()), max_value=int(df['Rotational speed'].max()))
tool_wear = st.slider('Износ инструмента', min_value=int(df['Tool wear'].min()), max_value=int(df['Tool wear'].max()))

# Подготовка входных данных
input_features = np.array([[air_temperature, process_temperature, rotational_speed, tool_wear]])

# Масштабируем данные
input_features_scaled = scaler.transform(input_features)

# Предсказание
if st.button('Прогнозировать'):
    prediction = model.predict(input_features_scaled)
    st.write(f'Предсказанное значение Torque: {prediction[0]:.2f}')
