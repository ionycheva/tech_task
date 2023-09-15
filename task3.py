import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import tensorflow as tf

'''Preprocessing of data: cleansing, casting to needed format,
    calculating required values and defining labels for classification.'''

# loading and preprocessing data
df = pd.read_excel('tmp.xlsx', header=1,)

# removing the ruble symbol and non-printing characters
df['Loan issued'] = pd.to_numeric(df['Loan issued'].str.replace('₽', '').str.replace('\u00A0', ''))
df['Earned interest'] = pd.to_numeric(df['Earned interest'].str.replace('₽', '').str.replace('\u00A0', ''))
df['Unpaid,  full amount'] = pd.to_numeric(df['Unpaid,  full amount'].str.replace('₽', '').str.replace('\u00A0', ''))

# casting some columns to percentage format
df['Comission, %'] = df['Comission, %'] * 100
df['EL'] = df['EL'] * 100

# column definition
df = df[['Comission, %', 'Rating', 'Loan issued', 'Earned interest', 'Unpaid,  full amount', 'EL']]

# calculation of required values
df['loss'] = df['Unpaid,  full amount'] * df['EL']
df['InvestorProfit'] = df['Earned interest'] - df['loss']
df['Commission'] = df['Loan issued'] * df['Comission, %'] / 100  # calculation of the absolute value of the commission
df['Profit'] = df['InvestorProfit'] + df['Commission']
df['Profit%'] = df['Profit'] / df['Loan issued']

# defining labels for classification
label1 = np.where(df['Profit%'] > 0)
df['label'] = 0
df['label'].iloc[label1] = 1


# Dataset making
X, Y_reg, Y_class = [], [], []

for i in df.index:
    X.append(df[['Comission, %', 'Rating', 'Loan issued', 'Earned interest', 'Unpaid,  full amount', 'EL']].iloc[i].values)
    Y_reg.append(df['Profit%'][i])
    Y_class.append(df['label'][i])

# Data scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# -----

'''Regression. Forecasting expected profit value.'''

# Slicing original dataset to train and test samples
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_reg, test_size=0.2, random_state=42)


# Defining model
inputs = keras.Input(shape=(6,1))
x = layers.BatchNormalization()(inputs)
x = layers.Conv1D(filters=20, kernel_size=(2), activation='relu')(x)
x = layers.Conv1D(filters=15, kernel_size=(2), activation='relu')(x)
x = layers.Conv1D(filters=10, kernel_size=(1), activation='relu')(x)
x = layers.Dense(250, activation='relu')(x)
x = layers.Dense(200, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="regression-model")
model.summary()

model.compile(loss=tf.keras.losses.LogCosh(),
              optimizer=tf.keras.optimizers.legacy.Adam(0.001),
              metrics=['mse', 'mae',])


# Model training
ephs = 28

history = model.fit(X_train, np.array(y_train), batch_size=8 , epochs=ephs, validation_split=0.3)
test_scores = model.evaluate(X_test, np.array(y_test), verbose=1)


# Show results
print('Regression results: ')
print("Test loss:", test_scores[0])
print("Test mse:", test_scores[1])
print("Test mae:", test_scores[2])
print('_______')


# -----

'''Classification. Predicts 1 when profit > 0, and 0 otherwise.'''

# Slicing original dataset to train and test samples
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_class, test_size=0.2, random_state=42)


# Defining model
inputs = keras.Input(shape=(6,1))
x = layers.BatchNormalization()(inputs)

x = layers.Conv1D(filters=5, kernel_size=(2), activation='relu')(x)
x = layers.Dense(300, activation='relu')(x)
x = layers.Dense(150, activation='relu')(x)
x = layers.Dense(75, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Flatten()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model_class = keras.Model(inputs=inputs, outputs=outputs, name="classification-model")
model_class.summary()

model_class.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.002),
    metrics=[ 'accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],)


# Model training
ephs = 16

history = model_class.fit(X_train, np.array(y_train), batch_size=8 , epochs=ephs, validation_split=0.2)
test_scores_CNN_class = model_class.evaluate(X_test, np.array(y_test), verbose=1)


# Show results
print('Classification results: ')
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
print("Test precision:", test_scores[2])
print("Test recall:", test_scores[3])









