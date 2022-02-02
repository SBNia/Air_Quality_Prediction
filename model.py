# loading dataset, plotting
import pandas as pd
import csv

# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

from config import *
from dataHelper import plotting_loss_chart, plotting_predictions_chart
from preProcess import create_model_data, data_labaling

def select_data(df, input_label):
    """
    Selects related features for a regression task
    Ex:
      For column_name = national
      Selects:
        year, month, day, hour
        national_+1, national_+2, national_+3, national_+4
    """

    # using year, month, day, hour and 4 shifts of data
    x_column, y_column = create_model_data(df, input_label)
    # Choose your test size to split between training and testing sets:
    x_train, x_test, y_train, y_test = train_test_split(x_column.values, y_column.values , test_size=0.16, random_state=42)

    # reshaping x, y
    x_train = x_train.reshape(x_train.shape + (1,))
    y_train = y_train.reshape( (len(y_train), 1) )
    x_test = x_test.reshape(x_test.shape + (1,))
    y_test = y_test.reshape( (len(y_test), 1) )

    return x_train, y_train, x_test, y_test


def train_model(input_label, x_train, y_train, x_test, y_test):
    """
    Creates an LSTM-based model
    Compiles it
    Trains it on training data and validates it on validation data
    Draws Loss chart

    Returns Model
    """

    # creating model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(x_train.shape[1], 1)))
    model.add(Dense(1))

    # compiling model
    model.compile(loss="mse", optimizer="adam")

    # fitting model on train data
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(x_test, y_test), shuffle=False)

    # plotting loss chart
    if PLOTTING == True: plotting_loss_chart(history, input_label)

    return model

def predict(model, input_label, x_test, y_test):

    # plot predictions
    predicted_value = model.predict(x_test)

    header = ['input_value', 'input_label', 'pred_value', 'pred_label']
    with open(RESULTS_DIR + MODE + '_' + input_label + '_result.csv', 'w', newline='',encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        for i in range(y_test.shape[0]):
            writer.writerow([y_test[i][0], data_labaling(y_test[i][0]), predicted_value[i][0], data_labaling(predicted_value[i][0])])

    if PLOTTING == True: plotting_predictions_chart(input_label, y_test, predicted_value)