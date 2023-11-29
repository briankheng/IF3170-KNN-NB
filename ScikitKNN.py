import csv
import math

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


train_data = pd.read_csv("data/data_train.csv")
# test_data.head()

# Variabel independen
x = train_data.drop(["price_range"], axis = 1)
# x.head()

# Variabel dependen
y = train_data["price_range"]
# y.head()

# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive Bayes
knn = KNeighborsClassifier(n_neighbors=15)
# Memasukkan data training pada fungsi klasifikasi Naive Bayes
knn.fit(x, y)

test_data = pd.read_csv("data/data_validation.csv")
x_test = test_data.drop(["price_range"], axis = 1)
y_actual = test_data["price_range"]

y_test = knn.predict(x_test)
print(classification_report(y_test, y_actual))