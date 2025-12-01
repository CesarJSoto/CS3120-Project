import pandas as pd
import numpy as np
import typing as t
import pickle
import json 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file_name = 'gen9vgc2025regj-1760.json'

class PokemonStats(t.TypedDict):
    players_using: float
    gxe_100th: float
    gxe_99th: float
    gxe_95th: float
    usage: float
    items: dict[str, float]
    moves: dict[str, float]
    types: dict[str, float]
    team: dict[str, float]



def normalize(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    return {k: v/total for k, v in d.items()}

def max_key(d: dict[str, float]) -> dict[str]:
        return max(d.keys(), key=lambda k: d[k])



with open(file_name) as file:
    raw: dict[str, dict[str, t.Any]] = json.load(file)["data"]

rows: list[PokemonStats] = []
for name, stats in raw.items():
    rows.append (PokemonStats(
        name = name,
        #players_using=stats["Viability Ceiling"][0],
        #gxe_100th=stats["Viability Ceiling"][1],
        #gxe_99th=stats["Viability Ceiling"][2],
        #gxe_95th=stats["Viability Ceiling"][3],
        #usage=stats["usage"],
        items=normalize(stats["Items"]),
        moves=normalize(stats["Moves"]),
        tera=max_key(stats["Tera Types"]),
        team=normalize(stats["Teammates"])
    ))
data = pd.json_normalize(rows).fillna(0.0)
data = data.set_index(data["name"].drop(columns=["name"]))

print(data)
'''
X = data.iloc[:,:-1]
y= data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifer = RandomForestClassifier
classifer.fit(X_train, y_train)
y_pred = classifer.predict(X_test)
'''