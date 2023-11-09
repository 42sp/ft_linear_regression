import sys
from my_data_frame import my_data_frame as mdf
from linear_regression import linear_regression

def main():
    model = linear_regression()
    try:
        mileage = float(input("Type the mileage im km: "))
    except ValueError as e:
        print(f"Error: {e}. You must input a valid integer or float!")
        return
    price_predicted = model.predict(mileage,standardize=True)
    print(f"The price for the mileage {mileage} is {price_predicted}")
    print(f"Theta0 = {model.theta0} and Theta1 = {model.theta1}")

if __name__ == "__main__":
    main()