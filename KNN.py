import csv
import math

# Load data from CSV file
def load_data(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            data = []
            for i in range(len(row)):
                if(i == 0 or i == 11 or i == 12 or i == 13 or i == 20): # TODO: Need to tune!
                    data.append(int(row[i]))
            dataset.append(data)
            
    return dataset

# Calculate Euclidean distance
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1): # TODO: Need to tune!
        if(i == 3): 
            distance += (row1[i] - row2[i])**2 * 1.5
        else:
            distance += (row1[i] - row2[i])**2 * 0.5
    return math.sqrt(distance)

# Get nearest neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
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

# Load and prepare data
train = load_data('data_train.csv')

# Define model parameter
num_neighbors = 1

# Test the KNN
test = load_data('data_validation.csv')

accuracy = test_knn(train, test, num_neighbors)
print('Accuracy: %.3f%%' % (accuracy))