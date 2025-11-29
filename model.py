import pandas as pd
import numpy as np
import sklearn as classification
import pickle

file = 'gen9vgc2025regj-1760.json'

df = pd.read_json(file)


print(df.head)
