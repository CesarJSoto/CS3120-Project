import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file = 'gen9vgc2025regj-1760.json'

data = pd.read_json(file)
print(f"** data has {data.shape[0]} rows and {data.shape[1]} columns **")



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 99)

