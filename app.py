
import os
import matplotlib
matplotlib.use('Agg')   # backend non-interactif pour Flask
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'cs-training.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

FEATURES = ['RevolvingUtilizationOfUnsecuredLines', 'age',
            'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

# libellés lisibles pour le formulaire
LIBELLES = {
    'RevolvingUtilizationOfUnsecuredLines': "Taux d'utilisation du credit (entre 0 et 1)",
    'age':                                  "Age du client (annees)",
    'NumberOfTime30-59DaysPastDueNotWorse': "Nombre de retards de 30 a 59 jours (2 dernieres annees)",
    'DebtRatio':                            "Ratio d'endettement (dettes / revenu)",
    'MonthlyIncome':                        "Revenu mensuel ($)",
    'NumberOfOpenCreditLinesAndLoans':      "Nombre de credits et prets en cours",
    'NumberOfTimes90DaysLate':              "Nombre de retards de 90 jours ou plus",
    'NumberRealEstateLoansOrLines':         "Nombre de prets immobiliers",
    'NumberOfTime60-89DaysPastDueNotWorse': "Nombre de retards de 60 a 89 jours",
    'NumberOfDependents':                   "Nombre de personnes a charge",
}


# chargement + imputation des NaN
df = pd.read_csv(DATA_PATH, index_col=0)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[['MonthlyIncome', 'NumberOfDependents']] = imputer.fit_transform(df[['MonthlyIncome', 'NumberOfDependents']])

y = df['SeriousDlqin2yrs']
X = df.drop('SeriousDlqin2yrs', axis=1)

# graphique distribution de la cible
plt.figure()
y.value_counts().plot.bar()
plt.title('Distribution de la variable cible')
plt.xlabel('SeriousDlqin2yrs')
plt.ylabel('Nombre')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'distribution_cible.png'))
plt.close()

# matrice de correlation
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr().round(2), annot=True)
plt.title('Matrice de correlation')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'correlation.png'))
plt.close()


# split + scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# entrainement des 6 modèles
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

print('Entrainement des 6 modeles (~2-3 min)...')

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_scaled, y_train)

dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

ada = AdaBoostClassifier(n_estimators=40, random_state=42)
ada.fit(X_train, y_train)

vc = VotingClassifier(estimators=[('logreg', LogisticRegression(max_iter=1000)),
                                  ('knn', KNeighborsClassifier(n_neighbors=8)),
                                  ('dt', DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1))], voting='soft')
vc.fit(X_train_scaled, y_train)

# (nom, modèle, X de test à utiliser)
modeles = [('Regression Logistique', logreg, X_test_scaled),
           ('KNN',                   knn,    X_test_scaled),
           ('Arbre de Decision',     dt,     X_test),
           ('Random Forest',         rf,     X_test),
           ('AdaBoost',              ada,    X_test),
           ('Voting Classifier',     vc,     X_test_scaled)]


# graphique importance des variables (random forest)
featimpor = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=["importance"]).sort_values(by="importance", ascending=True)
plt.figure(figsize=(8, 6))
plt.barh(featimpor.index, featimpor["importance"])
plt.xlabel("Importance")
plt.ylabel("Variables")
plt.title("Importance des variables (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'importance.png'))
plt.close()


# courbes ROC comparées + matrices de confusion par modèle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

AUCS = {}
CONFUSION_FILES = []

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], '--', color='gray')
for nom, m, X_ev in modeles:
    y_pred = m.predict(X_ev)
    y_pred_prob = m.predict_proba(X_ev)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    AUCS[nom] = round(float(auc), 3)
    plt.plot(fpr, tpr, label='{} (AUC = {:.3f})'.format(nom, auc))

    # matrice de confusion individuelle
    cm = confusion_matrix(y_test, y_pred, labels=m.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pas defaut', 'Defaut'])
    disp.plot()
    plt.title('Matrice de confusion - ' + nom)
    plt.tight_layout()
    filename = 'confusion_' + nom.lower().replace(' ', '_') + '.png'
    plt.savefig(os.path.join(STATIC_DIR, filename))
    plt.close()
    CONFUSION_FILES.append((nom, filename))

plt.xlabel('Taux faux positif')
plt.ylabel('Taux vrai positif')
plt.title('Courbes ROC - Comparaison des 6 modeles')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'roc_compare.png'))
plt.close()

# random forest = modèle utilisé pour la prédiction sur le site
medianes = X.median().to_dict()
print('Pret.')


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    resultat = None
    proba = None
    valeurs = {col: medianes[col] for col in FEATURES}

    if request.method == 'POST':
        for col in FEATURES:
            valeurs[col] = float(request.form[col])

        X_new = pd.DataFrame([valeurs])[FEATURES]
        X_new_scaled = scaler.transform(X_new)
        pred = rf.predict(X_new_scaled)[0]
        proba = round(float(rf.predict_proba(X_new_scaled)[0, 1]), 3)
        resultat = int(pred)

    return render_template('index.html', features=FEATURES, libelles=LIBELLES, valeurs=valeurs, resultat=resultat, proba=proba)


@app.route('/graphiques')
def graphiques():
    return render_template('graphiques.html', aucs=AUCS, confusion_files=CONFUSION_FILES)


if __name__ == '__main__':
    app.run(debug=True)
