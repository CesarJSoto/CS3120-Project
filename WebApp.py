# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

#Retrives model from pickle
model = pickle.load(open('my_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))
pokemon_names = pickle.load(open('names_list.pkl', 'rb'))

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
# ‘/’ URL is bound with hello_world() function.
@app.route('/', methods=["GET","POST"])
def web_app():
    if request.method == "POST":
        selected_pokemon = request.form.get("pokemon-choice")

        model.predict(selected_pokemon)

        return "You have selected: " + selected_pokemon + "\nwe recomend a tera type of:"
        + ""
    #tera_type = model.predict()
    return render_template("pokemon.html", pokemon_names=pokemon_names)
        


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)