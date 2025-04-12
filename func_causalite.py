import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

def sigmoid(x): # Définition de la fonction sigmoïde
    return 1 / (1 + np.exp(-x))

def regLogY(df):
    # Séparer les variables explicatives (X) et la cible (Y)
    X = df.drop(columns=['Y'])
    Y = df['Y']

    # Initialiser et entraîner le modèle
    model = LogisticRegression()
    model.fit(X, Y)

    X1 = X.copy()
    X1['Z'] = 1  # Mettre uniquement des 1 dans la colonne Z

    X0 = X.copy()
    X0['Z'] = 0  # Mettre uniquement des 0 dans la colonne Z

    # Prédictions
    Y1_pred = model.predict_proba(X1)[:,1]
    Y0_pred = model.predict_proba(X0)[:,1]

    tau_est = Y1_pred.mean()-Y0_pred.mean()
    return tau_est

def regArbY(df):
    # Séparer les variables explicatives (X) et la cible (Y)
    X = df.drop(columns=['Y'])
    Y = df['Y']

    # Initialiser et entraîner le modèle
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, Y)

    X1 = X.copy()
    X1['Z'] = 1  # Mettre uniquement des 1 dans la colonne Z

    X0 = X.copy()
    X0['Z'] = 0  # Mettre uniquement des 0 dans la colonne Z

    # Prédictions
    Y1_pred = model.predict_proba(X1)[:,1]
    Y0_pred = model.predict_proba(X0)[:,1]

    tau_est = Y1_pred.mean()-Y0_pred.mean()
    return tau_est

def regLogE(df):
    # Séparer les variables explicatives (X) et la cible (Y)
    X = df.drop(columns=['Y','Z'])
    Z = df['Z']
    Y = df['Y']
    
    # Initialiser et entraîner le modèle
    model = LogisticRegression()
    model.fit(X, Z)

    # Prédictions
    e_pred = model.predict_proba(X)[:,1]

    tau_est =  np.sum(Z * Y / e_pred) / np.sum(Z / e_pred) - np.sum((1-Z) * Y / (1-e_pred)) / np.sum((1-Z) / (1-e_pred))
    return tau_est, e_pred

def regArbE(df):
    # Séparer les variables explicatives (X) et la cible (Y)
    X = df.drop(columns=['Y','Z'])
    Z = df['Z']
    Y = df['Y']
    
    # Initialiser et entraîner le modèle
    model = DecisionTreeClassifier()
    model.fit(X, Z)

    # Prédictions
    e_pred = model.predict_proba(X)[:,1]

    tau_est =  np.sum(Z * Y / e_pred) / np.sum(Z / e_pred) - np.sum((1-Z) * Y / (1-e_pred)) / np.sum((1-Z) / (1-e_pred))
    return tau_est, e_pred

def genData(Nobs,alpha_tau,alpha_eZ,alpha_eY,alphaCross_eZ,alphaCross_eY):
    PX = 0.6 # probabilité qu'une covariable vaille 1
    NX = np.size(alpha_eZ) # nombre de covariables
    alpha0_eZ = -(np.sum(alpha_eZ) + np.sum(alphaCross_eZ)*PX) * PX #intercept du propensity score
    alpha0_eY = -(np.sum(alpha_eY) + +np.sum(alphaCross_eY)*PX + alpha_tau) * PX #intercept de la variable d'intérêt

    # GENERATION DES COVARIABLES X
    X = np.random.binomial(1,PX,(Nobs,NX))

    # EFFET CROISE POUR Z
    effCrossZ = np.zeros(Nobs)
    for i in range(NX):
        for j in range(NX):
            if alphaCross_eZ[i,j] != 0:
                effCrossZ += X[:,i] * X[:,j] * alphaCross_eZ[i,j]

    # GENERATION DE Z
    eZ = sigmoid(X.dot(alpha_eZ) + effCrossZ + alpha0_eZ) # Génération du propensity score
    Z = np.random.binomial(1,eZ)

    # EFFET CROISE POUR Y
    effCrossY = np.zeros(Nobs)
    for i in range(NX):
        for j in range(NX):
            if alphaCross_eY[i,j] != 0:
                effCrossY += X[:,i] * X[:,j] * alphaCross_eY[i,j]

    # GENERATION DE Y
    eY = sigmoid(X.dot(alpha_eY) + effCrossY + alpha_tau*Z + alpha0_eY)
    Y = np.random.binomial(1,eY)

    # Calcul de E(Y(1)) et E(Y(0))
    Z1 = np.ones(Nobs)
    Z0 = np.zeros(Nobs)

    EY1 = sigmoid(X.dot(alpha_eY) + effCrossY + alpha_tau*Z1 + alpha0_eY)
    EY0 = sigmoid(X.dot(alpha_eY) + effCrossY + alpha_tau*Z0 + alpha0_eY)

    # MISE EN FORME DES DONNEES
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(NX)])  # noms des colonnes X1, X2, ...
    df['Y'] = Y
    df['Z'] = Z
    df['EY1'] = EY1
    df['EY0'] = EY0

    # Réorganiser les colonnes pour avoir Y, Z, puis les X
    df = df[['Y', 'EY1', 'EY0', 'Z'] + [f"X{i+1}" for i in range(NX)]]

    # Calculer la moyenne de Y pour chaque valeur de Z
    mean_Y_by_Z = df.groupby('Z')['Y'].mean()

    # Afficher les résultats
    tau_pf = mean_Y_by_Z[1]-mean_Y_by_Z[0]  
    tau_causal = df['EY1'].mean()-df['EY0'].mean()

    return df, tau_pf, tau_causal