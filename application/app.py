
"""
Application de Prédiction de Maladie Cardiaque
"""


# Importe les packages.
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from flask import Flask, request, render_template
import sqlite3


# Instancie la classe Flask.
app = Flask(__name__)


# Charge le dataset coeur.
coeur = pd.read_excel("./dataset/coeur.xlsx")
df = coeur.copy()

# Normalise le dataset coeur.
for column in coeur.drop(["CŒUR"], axis=1).select_dtypes(np.number).columns:
    coeur[column] = coeur[column] / coeur[column].max()

# Encode le dataset coeur.
for column in coeur.drop(["CŒUR"], axis=1).select_dtypes("object").columns:
    coeur[column] = coeur[column].astype("category").cat.codes

# Répartis le dataset coeur.
x = coeur.drop("CŒUR", axis=1)
y = coeur["CŒUR"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Importe le modèle de régression logistique.
model = pickle.load(open("model.pkl", "rb"))


# Crée la base de données.
connection = sqlite3.connect("database.db")
cursor = connection.cursor()

# Crée la table user.
x_test.to_sql("user", connection, if_exists="replace", index=False)

connection.commit()
connection.close()


# Spécifie la route accueil.
@app.route("/")
def home():
    return render_template("home.html")


# Spécifie la route du formulaire.
@app.route("/form/", methods=["POST", "GET"])
def form():
    return render_template("simple_predict.html")


# Spécifie la route simple predict.
@app.route("/simple_predict/", methods=["POST", "GET"])
def simple_predict():
    try:
        # Enregistre les données du formulaire.
        data = {
            "AGE": int(request.form["age"]),
            "SEXE": request.form["sexe"],
            "TDT": request.form["tdt"],
            "PAR": int(request.form["par"]),
            "CHOLESTEROL": int(request.form["cholesterol"]), 
            "GAJ": int(str(request.form["gaj"])),
            "ECG": request.form["ecg"],
            "FCMAX": int(request.form["fcmax"]),
            "ANGINE": request.form["angine"],
            "DEPRESSION ": float(request.form["depression"]),
            "PENTE": request.form["pente"]
        }
    except:
        # Cas d'erreur.
        error = "Veuillez entrer que des valeurs numériques dans les champs concernés"
        return render_template("simple_predict.html", error=error)
    else:
        # Crée un dataframe.
        input_values = pd.DataFrame(data, index=[0])

        # Normalise les données du formulaire.
        for column in df.drop(["CŒUR"], axis=1).select_dtypes(np.number).columns:
            input_values[column] = input_values[column] / df[column].max()

        # Encode les données du formulaire.
        for column in df.drop(["CŒUR"], axis=1).select_dtypes("object").columns:
            input_values[column] = input_values[column].astype("category").cat.codes

        # Effectue la prédiction.
        prediction = model.predict(input_values)

        if prediction[0] == 0:
            retour = "Coeur sain"
        else:
            retour = "Coeur malade"

    return render_template("simple_predict.html", retour=retour)
    

# Spécifie la route multiple predict.
@app.route("/multiple_predict/", methods=["POST", "GET"])
def multiple_predict():
    # Crée la connection à la base de données.
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()

    if request.method == "POST":
        try:
            # Défini le nombre d'individu a prédire.
            individus = int(request.form["individu"])
        except:
            # Cas d'erreur.
            error = "Veuillez entrer une valeur numérique"
            return render_template("multiple_predict.html", error=error)
        else:
            # Selectionne le nombre d'individu a prédire.
            requete = cursor.execute(f"SELECT * FROM user ORDER BY RANDOM() LIMIT {individus}").fetchall()
            
            # Liste les données de chaque individu.
            row = [ligne for ligne in requete]

            liste = []

            for i in range(individus):
                # Transforme les données de chaque individu en tableau numpy.
                row[i] = list(row[i])
                features = np.array(row[i])
                features = features.reshape(1, -1)

                # Effectue la prédiction.
                prediction = model.predict(features)

                if prediction[0] == 0:
                    result = "Coeur sain"
                else:
                    result = "Coeur malade"

                # Ajoute le résultat de chaque prédiction.
                liste.append(result)
                # Défini le générateur d'itération.
                j = range(individus)

            connection.commit()
            connection.close()
            
        return render_template("multiple_predict.html", individus=individus, row=row, retour=liste, j=j)
    else:
        return render_template("multiple_predict.html")


# Exécute l'application.
if __name__ == "__main__":
    app.run(debug=True)
