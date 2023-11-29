from dataclasses import dataclass
from math import exp, pi, sqrt
import pandas as pd

@dataclass
class TrainingModel:
    price_count: list[int]
    categorical_probs: dict[tuple[str, str, int], float]
    mean: dict[tuple[str, int], float]
    std: dict[tuple[str, int], float]

# Function
def load_data(filename: str):
    return pd.read_csv(filename)

def accuracy_metric(actual: pd.DataFrame, predicted: pd.DataFrame):
    correct = 0
    for i in range(len(actual)):
        if actual[target_column][i] == predicted[target_column][i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def print_accuracy(test: pd.DataFrame, prediction: pd.DataFrame):
    accuracy = accuracy_metric(test, prediction)
    print("Accuracy: %.3f" % accuracy)

# Constant
sample_data = pd.read_csv("data/data_train.csv")
columns = sample_data.columns
categorical_columns = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi", "price_range"]
numerical_columns = [column for column in columns if column not in categorical_columns]
categorical_columns.pop() # remove price_range from category
target_column = "price_range"
price_range = [i for i in range(4)]

# Naive Bayes processing
def preprocess(data: pd.DataFrame):
    # Occurence for all price range
    price_count = [len(data[data[target_column] == i]) for i in range(4)]

    # for each price range, what is the prob for (category, value, price range) probability
    categorical_probs: dict[tuple[str, str, int], float] = dict()

    for category in categorical_columns:
        values: list[str] = list(set(data[category].values))
        for value in values:
            for price in price_range:
                filtered_data = data[(data[category] == value) & (data[target_column] == price)]
                count = len(filtered_data)
                prob = count / price_count[price]
                categorical_probs[(category, value, price)] = prob

    # statistic that will be used for calculating binomial naive bayes
    # dict value is dict[column, price range] : value
    mean: dict[tuple[str, int], float] = dict()
    std: dict[tuple[str, int], float] = dict()
    for column in numerical_columns:
        for price in price_range:
            filtered_data = data[data[target_column] == price][column]
            mean[(column, price)] = filtered_data.mean()
            std[(column, price)] = filtered_data.std()
    
    model = TrainingModel(price_count, categorical_probs, mean, std)
    return model

def count_binomial(value: float, mean: float, std: float):
    exponent = pow(value - mean, 2) / (2 * pow(std, 2))
    exp_value = exp(-exponent)
    denominator = std * sqrt(2 * pi)
    binomial = exp_value / denominator
    return binomial

def train_data(data: pd.DataFrame, model: TrainingModel):
    price_predictions: list[int] = []
    for i in range(len(data)):
        maximum_prob = 0
        max_prob_price = -1
        for price in price_range:
            # Initial probability
            price_prob = model.price_count[price] / len(data)
            prob = price_prob
            
            # Categorical probability
            for category in categorical_columns:
                category_prob = model.categorical_probs[(category, data[category][i], price)]
                prob *= category_prob

            # Numerical probability
            for column in numerical_columns:
                tuple_data = (column, price)
                numerical_prob = count_binomial(data[column][i], model.mean[tuple_data], model.std[tuple_data])
                prob *= numerical_prob

            if prob > maximum_prob:
                maximum_prob = prob
                max_prob_price = price

        price_predictions.append(max_prob_price)
    
    training_result[target_column] = price_predictions

if __name__ == '__main__':
    train = load_data("data/data_train.csv")
    model = preprocess(train)
    
    real_data = load_data("data/data_validation.csv")
    training_result = load_data("data/data_validation.csv")
    train_data(training_result, model)
    
    print_accuracy(real_data, training_result)