# plotting

import pandas as pd
import matplotlib.pyplot as plt

from config import *

def plotting_loss_chart(history, input_label):

    plt.figure(figsize=(20, 15))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper left')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(PRESENTATION_DIR+MODE+'_plotting_loss_chart[' + input_label + '].png')

def plotting_predictions_chart(input_label, input_value, predicted_value):

    index = list(range(len(input_value)))
    plt.figure(figsize=(20, 15))
    plt.plot(index, input_value, marker=".", label="input value")
    plt.plot(index, predicted_value, label="predicted value")
    plt.title('Input vs Predicted Data in '+MODE+' mode')
    plt.xlabel("time")
    plt.ylabel("PSI")
    plt.legend(loc='upper right')
    plt.savefig(PRESENTATION_DIR+MODE+'_plotting_predictions_chart[' + input_label + '].png')
