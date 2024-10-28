# %% [Importando bibliotecas]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %% [Carregar dados]
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")

# %% [Preenchendo valores faltantes]
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].value_counts().idxmax())
train["CabinKnown"] = train["Cabin"].notnull().astype(int)

# Convertendo 'Sex' para numérico
train["Gender"] = train["Sex"].map({"male": 0, "female": 1})
train = pd.get_dummies(train, columns=["Embarked"], prefix="Embarked", drop_first=True)

# %% [Criar variáveis adicionais]
train["Tittle"] = train["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
common_tittles = ["Mr", "Mrs", "Miss", "Master"]
train["Tittle"] = train["Tittle"].apply(
    lambda x: x if x in common_tittles else "Others"
)
train = pd.get_dummies(train, columns=["Tittle"], prefix="Tittle", drop_first=True)
train["FamilySize"] = train["SibSp"] + train["Parch"]
train["IsAlone"] = (train["FamilySize"] == 0).astype(int)

# Normalizando 'Fare'
scaler = StandardScaler()
train["Fare"] = scaler.fit_transform(train[["Fare"]])

# %% [Definindo as features e a variável alvo]
X = train.drop(["Survived", "Name", "Ticket", "Cabin", "Sex"], axis=1)
y = train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [Criando um pipeline para Random Forest]
pipeline = Pipeline([("classifier", GradientBoostingClassifier())])
cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(
    f"Acurácia média com validação cruzada: {cross_val_scores.mean():.2f} +/- {cross_val_scores.std():.2f}"
)

# Treinando e avaliando o modelo
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
print(f"Acurácia: {accuracy_score(y_val, y_pred):.2f}")
print(classification_report(y_val, y_pred))

# %% [Preparar o conjunto de teste]
test["Gender"] = test["Sex"].map({"male": 0, "female": 1})
test["CabinKnown"] = test["Cabin"].notnull().astype(int)
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna(test["Embarked"].value_counts().idxmax())
test = pd.get_dummies(test, columns=["Embarked"], prefix="Embarked", drop_first=True)

test["Tittle"] = test["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
test["Tittle"] = test["Tittle"].apply(lambda x: x if x in common_tittles else "Others")
test = pd.get_dummies(test, columns=["Tittle"], prefix="Tittle", drop_first=True)

test["FamilySize"] = test["SibSp"] + test["Parch"]
test["IsAlone"] = (test["FamilySize"] == 0).astype(int)
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Fare"] = scaler.transform(test[["Fare"]])

test_final = test.drop(["Name", "Ticket", "Cabin", "Sex"], axis=1)
test_final = test_final.reindex(columns=X.columns, fill_value=0)

# Fazendo previsões e criando o arquivo de submissão
predictions = pipeline.predict(test_final)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
submission.to_csv("submission.csv", index=False)
print("Arquivo de submissão salvo como 'submission.csv'.")

# %%
