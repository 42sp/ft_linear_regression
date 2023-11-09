import sys
import numpy as np
import matplotlib.pyplot as plt
from my_data_frame import my_data_frame as mdf
from linear_regression import linear_regression

def main():
    if len(sys.argv) != 2:
        print("Usage: python bonus.py data.csv")
        return
    file_name = sys.argv[1]
    df = mdf(file_name)
    model = linear_regression()
    y_pred = [model.predict(x, standardize=True) for x in df.data['km']]
    x_reg = np.linspace(min(df.data['km']), max(df.data['km']), 100)
    y_reg = [model.predict(x, standardize=True) for x in x_reg]

    plt.scatter(x_reg, y_reg, label='regression', color='blue')
    plt.scatter(df.data['km'], y_pred, label='real data', color='red')
    plt.annotate(f'MSE: {model.mse:.4f}', xy=(0.7, 0.4), xycoords='axes fraction', fontsize=12)
    plt.xlabel('Milage [km]')
    plt.ylabel('Price $') 
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()