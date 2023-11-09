import sys
from my_data_frame import my_data_frame as mdf
from linear_regression import linear_regression

def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py data.csv")
        return
    file_name = sys.argv[1]
    df = mdf(file_name)
    model = linear_regression()
    model.train(df.data['km'], df.data['price'])

if __name__ == "__main__":
    main()