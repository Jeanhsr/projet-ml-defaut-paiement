# Projet ML — Prédiction de défaut de paiement bancaire

## Structure du projet

```
rendu_final/
├── app.py                  # Application Flask principale
├── requirements.txt        # Bibliothèques Python nécessaires
├── Procfile                # Commande de démarrage pour Render
├── .gitignore
├── data/
│   └── cs-training.csv     # Dataset (à ajouter manuellement)
├── templates/
│   ├── index.html          # Page de prédiction
│   └── graphiques.html     # Page des graphiques
└── static/                 # Graphiques générés automatiquement
```

## Déploiement sur Render

### Étape 1 — Mettre le projet sur GitHub
1. Crée un compte sur github.com
2. Crée un nouveau repository (New repository)
3. Upload tous les fichiers du projet
4. ⚠️ N'oublie pas de mettre `cs-training.csv` dans le dossier `data/`

### Étape 2 — Déployer sur Render
1. Crée un compte sur render.com
2. Clique sur "New Web Service"
3. Connecte ton repository GitHub
4. Render détecte automatiquement le Procfile
5. Clique sur "Deploy"

### Étape 3 — Attendre le déploiement
- Le premier déploiement prend 5-10 minutes
- L'entraînement des modèles se fait au démarrage (~2-3 min)
- Ton site sera accessible à l'URL fournie par Render

## Lancer en local
```bash
pip install -r requirements.txt
python app.py
```
Puis ouvrir http://127.0.0.1:5000
