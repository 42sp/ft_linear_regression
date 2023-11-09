import random
import csv
import os
import matplotlib.pyplot as plt

class linear_regression:
  def __init__(self, epochs = 1000, lr = 0.01, test_ratio = 0.2) -> None:
    self.epochs = epochs
    self.lr = lr
    self.test_ratio = test_ratio
    self.load('params.csv')
    self.mse_lst = []
  
  def split_train_test(self, x, y):
    combined = list(zip(x, y))
    random.shuffle(combined)
    shuffled_x, shuffled_y = zip(*combined)

    split_point = int((1 - self.test_ratio) * len(shuffled_x))
    self.x_train = shuffled_x[:split_point]
    self.y_train = shuffled_y[:split_point]
    self.x_test = shuffled_x[split_point:]
    self.y_test = shuffled_y[split_point:]

  def standardize(self, x,y):
    self.min_x = min(x)
    self.max_x = max(x)
    self.min_y = min(y)
    self.max_y = max(y)
    x = [(a - self.min_x)/(self.max_x - self.min_x) for  a in x]
    y = [(b - self.min_y)/(self.max_y - self.min_y) for  b in y]
    return(x,y)

  def train(self, x, y):
    x, y = self.standardize(x,y)
    self.split_train_test(x, y)
    m = len(self.x_train)
    for epoch in range(self.epochs):
      t0 = self.lr * (sum([self.predict(a) - b for a,b in zip(self.x_train, self.y_train) ]) /m )
      t1 = self.lr * (sum([(self.predict(a) - b)*a for a,b in zip(self.x_train, self.y_train) ]) /m)
      self.theta0 -= t0
      self.theta1 -= t1
      self.mse_lst.append((epoch, self.calculate_mse()))
      self.mse = self.mse_lst[-1][1]
    self.plot_ms2()
    self.save('params.csv')

  def predict(self,x, standardize=False):
    if standardize and self.theta0 != 0 and self.theta1 != 0:
      standardize_x = (x - self.min_x)/(self.max_x - self.min_x)
      standardize_y = self.theta0 + (self.theta1*standardize_x)
      return standardize_y * (self.max_y - self.min_y) + self.min_y
    else:
      return self.theta0 + (self.theta1*x)
  
  def calculate_mse(self):
    test = sum([(self.predict(a) - b)**2 for a,b in zip(self.x_test, self.y_test)])/len(self.x_test)
    return test

  def plot_ms2(self):
    epochs, ms2_values = zip(*self.mse_lst)
    plt.plot(epochs, ms2_values)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Mean Square Error vs. Epoch")
    plt.show()

  def save(self, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['theta0', 'theta1', 'min_x', 'min_y', 'max_x', 'max_y', 'mse'])
        writer.writerow([self.theta0, self.theta1, self.min_x, self.min_y, self.max_x, self.max_y, self.mse])

  def load(self, filename):
    if os.path.exists(filename):
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader) 
            row = next(reader)
            self.theta0 = float(row[0])
            self.theta1 = float(row[1])
            self.min_x = float(row[2])
            self.min_y = float(row[3])
            self.max_x = float(row[4])
            self.max_y = float(row[5])
            self.mse =  float(row[6])
            print("Parameters loaded from", filename)
    else:
        print("File", filename, "does not exist. Initializing theta0 and theta1 to 0.")
        self.theta0 = 0
        self.theta1 = 0
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        self.mse = 0