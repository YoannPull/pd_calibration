# Credit Risk Modeling Pipeline (PD Model)

**Bank-Grade Probability of Default (PD) Modeling Pipeline**

Ce projet implémente un pipeline complet de modélisation du risque de crédit en Python. À partir des données brutes (prêts hypothécaires), il génère des variables cibles, discrétise les variables (Binning Monotone), les transforme en WOE (Weight of Evidence), entraîne une régression logistique calibrée et génère une grille de notation (Master Scale).

L'architecture se concentre sur la **robustesse temporelle**, l'**interprétabilité** et la **prévention du Data Leakage** (fuite de données).

---

## Architecture du Projet

```plaintext
.
├── Makefile                   # Orchestrateur principal (Entry Point)
├── config.yml                 # Configuration (chemins, fenêtres temporelles)
├── pyproject.toml             # Dépendances (Poetry)
├── README.md                  # Documentation
├── src/                       # Code Source
│   ├── make_labels.py         # Génération des labels (default_24m) & Vintage
│   ├── impute_and_save.py     # Imputation stricte (Fit Train / Transform Val)
│   ├── fit_binning.py         # Binning Monotone (Max Gini)
│   ├── train_model.py         # Sélection des features, LR, Scaling, Calibration
│   ├── apply_model.py         # Scoring & Application (Mode production)
│   ├── generate_report.py     # Reporting HTML (Train vs Validation)
│   └── features/              # Modules utilitaires (binning, etc.)
├── data/
│   ├── raw/                   # Données brutes
│   └── processed/             # Données intermédiaires et scorées
└── artifacts/                 # Modèles sérialisés et Grilles
    ├── imputer/               # imputer.joblib
    ├── binning_maxgini/       # Bins.json
    └── model_from_binned/     # model.joblib, Risk_buckets.json
```

---

## Démarrage Rapide

### Prérequis

* Python 3.9+
* Poetry (Gestionnaire de dépendances)
* Make

### Installation

```bash
# Installer les dépendances
poetry install
```

### Exécution du Pipeline Complet

Pour exécuter toutes les étapes (génération des labels, imputation, binning, entraînement et reporting), utilisez la commande suivante :

```bash
make pipeline
```

---

## Détail des Étapes (Workflow)

Le pipeline est séquencé via le `Makefile` pour garantir la reproductibilité.

### 1. Génération des Labels (`make labels`)

* Construction de la variable cible `default_24m` (Défaut dans les 24 mois suivant l'origination).
* Split temporel strict :

  * **Train** : Vintages 2014-2018 (Données matures).
  * **Validation** : Vintages 2019-2020 (Test Out-of-Time).

### 2. Imputation (`make impute`)

* Remplacement des valeurs manquantes.
* **Sécurité** : L'imputer est "fitté" uniquement sur le Train. Les statistiques (médiane/mode) sont sauvegardées et appliquées à la Validation pour éviter toute fuite d'information.

### 3. Binning & WOE (`make binning_fit`)

* Discrétisation des variables continues (ex : `credit_score`, `dti`) en 5 à 10 bins.
* **Monotonicité forcée** : Les bins doivent avoir une tendance de taux de défaut strictement monotone (croissante ou décroissante).
* Transformation en WOE (Weight of Evidence).

### 4. Entraînement (`make model_train`)

* **Feature Selection** : Suppression des variables trop corrélées (> 0.85).
* **Anti-Leakage (Blacklist)** : Exclusion stricte des variables temporelles (quarter, vintage), des dates explicites (maturity_date) et des indicateurs futurs (mi_cancellation).
* **Modélisation** : Régression Logistique avec pénalité L2 (Ridge).
* **Scaling** : Conversion des Log-Odds en Points (Score). Base : 600 points. PDO (Points to Double Odds) : 20.
* **Calibration** : Régression isotonic pour aligner la PD prédite sur la PD observée.
* **Master Scale** : Création d'une grille de notation (Grades 1 à 10) basée sur les quantiles du Train.

### 5. Reporting (`make report`)

* Génération de rapports HTML pour le Train et la Validation.
* **Métriques** : AUC, Gini, LogLoss, Brier Score.
* **Graphiques** : Courbe ROC, Calibration Plot, Distribution des Scores.
* **Master Scale Analysis** : Vérification de la stabilité des volumes et des PD par grade.
* **Model Specs** : Affichage des coefficients et de l'intercept.

---

## Mécanismes de Sécurité (Bank-Grade)

Ce pipeline intègre plusieurs contrôles pour éviter le sur-apprentissage et le *Population Shift* :

* **Blacklist Variables** : Exclusion automatique des métadonnées techniques (ex : `__file_quarter`) et des variables temporelles explicites ou proxies (quarter, property_valuation_method).
* **Frozen Artifacts** : Les bornes de binning, les mappages WOE et la grille de notation sont calculés sur le Train et "gelés" (sauvegardés en JSON/Joblib), puis appliqués à la Validation.
* **Monotonic Constraints** : Interdiction des relations non-linéaires complexes (ex : "U-shape") qui sont souvent instables dans le temps.

---

## Commandes Utilitaires

* **Scorer un fichier externe (OOS)** :

```bash
make oos_score
```

* **Scorer un fichier spécifique (Custom)** :

```bash
make score_custom DATA=path/to/file.parquet OUT=path/to/output.parquet
```

* **Nettoyer le projet** :

```bash
make clean_all
```

---

## Auteurs & Licence

Ce projet a été développé pour la calibration et validation de modèles PD réglementaires.

* **Licence** : Propriétaire / Interne.