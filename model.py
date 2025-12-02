#Data stuff
import pandas as pd
import numpy as np
import typing as t
import pickle
import json 

#Model creation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#Libraries for Visualizing tree
from sklearn.tree import export_graphviz
import graphviz


file_name = 'gen9vgc2025regj-1760.json'
dot_file = 'tree.dot'
NUM_ESTIMATORS = 80


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
    counter: dict[str, float]


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

le = LabelEncoder()
data["name"] = le.fit_transform(data["name"])
data["tera_encoded"] = le.fit_transform(data["tera"])


X = data.drop(["name","tera","tera_encoded"],axis=1)
y = data["tera_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifer = RandomForestClassifier(n_estimators=NUM_ESTIMATORS,random_state=1)
classifer.fit(X_train, y_train)
y_pred = classifer.predict(X_test)

print(X_test)
print(le.inverse_transform(y_pred))

tree = classifer.estimators_[NUM_ESTIMATORS - 1]
export_graphviz(tree, out_file=dot_file,
                class_names=data["tera"],
                rounded=True,filled=True)

with open(dot_file) as f:
     dot_graph = f.read()

graph = graphviz.Source(dot_graph)
graph.render("randm_forest_tree")
graph