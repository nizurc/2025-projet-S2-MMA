import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

#-----------------------------------------------------------
def sigmoid(x): # Définition de la fonction sigmoïde
    return 1 / (1 + np.exp(-x))

#-----------------------------------------------------------
def regLogY(df,vars_to_include = None): # régréssion logistique OR
    # Séparer les variables explicatives (X) et la cible (Y)
    Y = df['Y']
    if vars_to_include is None:
        X = df.drop(columns=['Y','EY0','EY1'])
    else:
        X = df[vars_to_include + ['Z']]

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

#-----------------------------------------------------------
def regArbY(df,vars_to_include = None):
    # Séparer les variables explicatives (X) et la cible (Y)
    Y = df['Y']
    if vars_to_include is None:
        X = df.drop(columns=['Y','EY0','EY1'])
    else:
        X = df[vars_to_include + ['Z']]

    # Initialiser et entraîner le modèle
    model = DecisionTreeClassifier()
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

#-----------------------------------------------------------
def regLogE(df,vars_to_include = None):
    # Séparer les variables explicatives (X) et la cible (Y)
    Z = df['Z']
    Y = df['Y']
    if vars_to_include is None:
        X = df.drop(columns=['Y','Z','EY0','EY1'])
    else:
        X = df[vars_to_include]
    
    # Initialiser et entraîner le modèle
    model = LogisticRegression()
    model.fit(X, Z)

    # Prédictions
    e_pred = model.predict_proba(X)[:,1]

    tau_est =  np.sum(Z * Y / e_pred) / np.sum(Z / e_pred) - np.sum((1-Z) * Y / (1-e_pred)) / np.sum((1-Z) / (1-e_pred))
    return tau_est, e_pred

#-----------------------------------------------------------
def regArbE(df,vars_to_include = None):
    # Séparer les variables explicatives (X) et la cible (Y)
    Z = df['Z']
    Y = df['Y']
    if vars_to_include is None:
        X = df.drop(columns=['Y','Z','EY0','EY1'])
    else:
        X = df[vars_to_include]
    
    # Initialiser et entraîner le modèle
    model = DecisionTreeClassifier()
    model.fit(X, Z)

    # Prédictions
    e_pred = model.predict_proba(X)[:,1]

    tau_est =  np.sum(Z * Y / e_pred) / np.sum(Z / e_pred) - np.sum((1-Z) * Y / (1-e_pred)) / np.sum((1-Z) / (1-e_pred))
    return tau_est, e_pred

#-----------------------------------------------------------
def genData(Nobs,alpha_tau,alpha_eZ,alpha_eY,alphaXCrossX_eZ,alphaXCrossX_eY,alphaXCrossZ_eY):
    PX = 1/3 # probabilité qu'une covariable vaille 1
    NX = np.size(alpha_eZ) # nombre de covariables
    alpha0_eZ = -(np.sum(alpha_eZ) + np.sum(alphaXCrossX_eZ)*PX) * PX #intercept du propensity score
    alpha0_eY = -(np.sum(alpha_eY) + +np.sum(alphaXCrossX_eY)*PX + alpha_tau) * PX #intercept de la variable d'intérêt

    # GENERATION DES COVARIABLES X
    X = np.random.binomial(1,PX,(Nobs,NX))

    # EFFET CROISE POUR Z
    effCrossZ = np.zeros(Nobs)
    for i in range(NX):
        for j in range(NX):
            if alphaXCrossX_eZ[i,j] != 0:
                effCrossZ += X[:,i] * X[:,j] * alphaXCrossX_eZ[i,j]

    # GENERATION DE Z
    eZ = sigmoid(X.dot(alpha_eZ) + effCrossZ + alpha0_eZ) # Génération du propensity score
    Z = np.random.binomial(1,eZ)

    # EFFET CROISE XxX POUR Y
    effCrossY1 = np.zeros(Nobs)
    for i in range(NX):
        for j in range(NX):
            if alphaXCrossX_eY[i,j] != 0:
                effCrossY1 += X[:,i] * X[:,j] * alphaXCrossX_eY[i,j]
    
    # EFFET CROISE XxZ POUR Y
    effCrossY2 = np.zeros(Nobs)
    for i in range(NX):
            effCrossY2 += X[:,i] * Z * alphaXCrossZ_eY[i]

    # GENERATION DE Y
    eY = sigmoid(X.dot(alpha_eY) + effCrossY1 + effCrossY2 + alpha_tau*Z + alpha0_eY)
    Y = np.random.binomial(1,eY)

    # Calcul de E(Y(1)) et E(Y(0))
    Z1 = np.ones(Nobs)
    Z0 = np.zeros(Nobs)

    EY1 = sigmoid(X.dot(alpha_eY) + effCrossY1 + effCrossY2 + alpha_tau*Z1 + alpha0_eY)
    EY0 = sigmoid(X.dot(alpha_eY) + effCrossY1 + effCrossY2 + alpha_tau*Z0 + alpha0_eY)

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

#-----------------------------------------------------------
def print_moy_strat(df):
    # Regrouper par (X1, X2, Z) et calculer la moyenne de Y et l'effectif
    grouped = df.groupby(['X1', 'X2', 'X3', 'X4', 'Z']).agg(
    mean_Y=('Y', 'mean'),
    count=('Y', 'count')
    ).reset_index()

    # Réorganiser les résultats pour avoir les colonnes séparées pour Z=0 et Z=1
    pivoted = grouped.pivot(index=['X1', 'X2', 'X3', 'X4'], columns='Z', values=['mean_Y', 'count'])

    # Renommer les colonnes pour plus de clarté
    pivoted.columns = ['E[Y|Z=0]', 'E[Y|Z=1]', 'N(Z=0)', 'N(Z=1)']
    pivoted = pivoted.fillna(0)  # Remplir les valeurs NaN si une combinaison (X1, X2, Z) est absente

    # Calcul de la différence E[Y | Z=1] - E[Y | Z=0]
    pivoted['Diff'] = pivoted['E[Y|Z=1]'] - pivoted['E[Y|Z=0]']

    # Calcul du poids total N(Z=0) + N(Z=1)
    pivoted['Poids'] = pivoted['N(Z=0)'] + pivoted['N(Z=1)']

    # Calcul de la moyenne pondérée
    tau_est1 = (pivoted['Diff'] * pivoted['Poids']).sum() / pivoted['Poids'].sum()

    # Affichage des résultats
    print("RESULTATS DE LA MOYENNE STRATIFIÉE :")
    print(pivoted)
    print("tau_estimé (OR moy. strat) =", round(tau_est1, 3))

#-----------------------------------------------------------
def print_res_sim(df,vars_to_includeY=None,vars_to_includeE=None):
    tau_est2 = regLogY(df,vars_to_includeY)
    tau_est3 = regArbY(df,vars_to_includeY)
    tau_est4, e_pred = regLogE(df,vars_to_includeE)
    tau_est5, e_pred = regArbE(df,vars_to_includeE)
    print("RESULTATS DES DIFFERENTS MODELES :")
    print("tau_estimé (IPW reg.log.) =", round(tau_est4, 3))
    print("tau_estimé (IPW arb.dec.) =", round(tau_est5, 3))
    print("tau_estimé (OR reg. log.) =", round(tau_est2, 3))
    print("tau_estimé (OR arb. dec.) =", round(tau_est3, 3))

    # Tracer l'histogramme propensity score
    #plt.hist(e_pred, bins=10, edgecolor='black', range=(0, 1))  # 'bins' définit le nombre de bacs dans l'histogramme

    # Ajouter des labels et un titre
    #plt.xlabel('Valeurs')
    #plt.ylabel('Fréquence')
    #plt.title('Répartition des valeurs du propensity score prédit')

    # Afficher l'histogramme
    #plt.show()