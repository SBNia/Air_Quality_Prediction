from keras.models import load_model

from preProcess import creating_lagged_variables, \
                        load_data,\
                        preprocessing_data, \
                        create_model_data
from model import select_data, train_model, predict
from config import *

# Defining main function
def main():

    df = load_data()
    pre_proc_df = preprocessing_data(df)
    df_with_lag = creating_lagged_variables(pre_proc_df, COLUMNS, 4)
    df_with_lag.dropna(inplace=True)

    if MODE == "TRAIN":
        for item in COLUMNS:
            x_train, y_train, x_test, y_test = select_data(df_with_lag, item)
            model = train_model(item, x_train, y_train, x_test, y_test)
            model.save(MODEL_DIR+'lstm_'+item+'_model.h5')
            predict(model, item, x_test, y_test)
    else:
        model = load_model(MODEL_DIR + 'lstm_'+MODEL_NAME+'_model.h5')

        x_column, y_column = create_model_data(df_with_lag, MODEL_NAME)
        x_test = x_column.values.reshape(x_column.values.shape + (1,))
        y_test = y_column.values.reshape((len(y_column.values), 1))

        predict(model, MODEL_NAME, x_test, y_test)

# Using the special variable
if __name__ == "__main__":
    main()
