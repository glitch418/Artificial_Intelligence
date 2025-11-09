import sys
import numpy as np
import csv
import matplotlib.pyplot as plt

def main():
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])
    x, y = load_data(filename)
    normalized_x = normalize_data(x)
    w, b = closed_form_solution(normalized_x, y)
    #Q1
    #SUBMIT hw5.csv!!!
    #Q2
    plot_years_vs_days(x, y)

    #Q3
    print("Q3:")
    print(normalized_x)
    
    #Q4
    print("Q4:")
    print(np.array([w, b]))

    #Q5
    print("Q5a:")
    gradient_descent(normalized_x, y, learning_rate, iterations)
    print("Q5b: 0.7")
    print("Q5c: 400")

    #Q6
    print("Q6: " + str(prediction(2023, x, w, b)))

    #Q7
    symbol, interpretation =  interpret_weight_sign(w)
    print("Q7a: " + symbol)
    print("Q7b: " + interpretation)

    #Q8
    year_no_freeze = str(predict_no_freeze(x, w, b))
    interpretation = (
    "This model predicts that the lake will stop freezing by the year 2463. It tells us that the climate is getting warmer"
    " as number of frozen ice days are decreasing. However, it is not compelling as there is a high level of uncertainty associated with"
    " it such as climate change and human activity. This model tries to establish a linear relationship between year and number of "
    "frozen ice days which is incorrect as many other variables are in play. Therefore, neglecting other variables and relying on a"
    " single dataset without considering global events happening around that year cause this skewed prediction. This is therefore "
    "a potential limitation of our model."
)
    print("Q8a: " + year_no_freeze)
    print("Q8b: " + interpretation)


def load_data(filename):
    try:
        years = []
        days = []
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                years.append(float(row['year']))
                days.append(float(row['days']))
        return np.array(years), np.array(days)
    except Exception as e:
        return("Error loading the dataset")
    
def plot_years_vs_days(x, y):
    plt.figure()
    plt.plot(x, y, linestyle='-', color='b')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.title('Year vs. Number of Frozen Days')
    plt.savefig("data_plot.jpg")
    plt.clf()

#returns matrix of dim { n x 2}
def normalize_data(x):
    min_value = np.min(x)
    max_value = np.max(x)
    normalized_x = (x - min_value) / (max_value - min_value)
    n = len(normalized_x)
    augmented_features = np.vstack((normalized_x, np.ones(n))).T
    return augmented_features

#returns matrix of dim { 1 x 2}
def closed_form_solution(x_normalized, y):
    weights = np.dot(np.linalg.inv(np.dot(x_normalized.T, x_normalized)), np.dot(x_normalized.T, y))
    return weights

def gradient_descent(normalized_x, y, alpha, iterations):
    w = 0.0
    b = 0.0
    losses = []
    n = len(y)
    x_aug = normalized_x
    for i in range(iterations):
        if i % 10 == 0:
            print(np.array([w, b]))
        #x_aug{n x 2}    np.array([w, b]){1 x 2}    y_pred-y{1 x n}    C{1 x 2}
        y_pred = np.dot(np.array([w, b]), x_aug.T)
        C = (1/n) * np.dot((y_pred - y), x_aug)
        w -= alpha * C[0]
        b -= alpha * C[1]
        #calculating loss for plotting
        mse_loss = (1 / (2 * n)) * np.sum(np.square(y_pred - y))
        losses.append(mse_loss)
    
    plt.figure()
    plt.plot(losses, color='b')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss over Iterations')
    plt.savefig("loss_plot.jpg")
    plt.clf()
    return np.array([w, b])
    
def prediction(year, x, w, b):
    min_value = np.min(x)
    max_value = np.max(x)
    normalized_input = (year - min_value) / (max_value - min_value)
    y_hat = (w * normalized_input) + b
    return y_hat

def interpret_weight_sign(w):
    symbol = None
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="
    interpretation = ("If w > 0, it shows that the number of frozen ice days for the lake increases as years increase."
     "If w < 0, it shows that the number of frozen ice days for the lake decreases as years increase which might suggest warmer climate."
     "If w = 0, it shows that there is no significant relationship between the number of frozen ice days and years.")
    return symbol, interpretation

def predict_no_freeze(x, w, b):
    min_value = np.min(x)
    max_value = np.max(x)
    year = (((-b) * (max_value - min_value))  *  (1 / w))     +     min_value   
    return year

if __name__ == "__main__":
    main()