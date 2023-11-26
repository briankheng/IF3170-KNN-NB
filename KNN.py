import csv
import math

# Load data from CSV file
def load_data(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            for i in range(len(row)-1):
                row[i] = float(row[i]) * feature_selection[i]
            dataset.append(row)
    return dataset

# Normalize the data
def normalize_data(dataset):
    minmax = []
    for i in range(len(dataset[0])-1):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    for row in dataset:
        for i in range(len(row)-1):
            if minmax[i][1] - minmax[i][0] != 0:
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Calculate Euclidean distance
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2 * feature_weights[i]
    return math.sqrt(distance)

# Get nearest neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a prediction
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Test the KNN on the testing set
def test_knn(train, test, num_neighbors):
    predictions = []
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predictions)
    return accuracy

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Define feature selection
#                    0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19
feature_selection = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Define feature weights
#                  0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19
feature_weights = [1.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.5, 1.5, 7.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Load and prepare data
train = load_data('data/data_train.csv')
normalize_data(train)

# Define model parameter
num_neighbors = 15

# Test the KNN
test = load_data('data/data_validation.csv')
normalize_data(test)

accuracy = test_knn(train, test, num_neighbors)
print('Accuracy: %.3f%%' % accuracy)