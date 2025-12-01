# Credit Risk Modeling Pipeline (PD Model)

**Bank-Grade Probability of Default (PD) Modeling & Calibration Pipeline**

Ce projet implémente un pipeline complet de **modélisation du risque de crédit** en Python.
À partir de données brutes (prêts hypothécaires), il :

1. génère des labels de défaut (ex : `default_24m`),
2. applique une imputation stricte (anti-leakage),
3. construit un binning monotone (Max |Gini|),
4. transforme les features en WOE,
5. entraîne une régression logistique avec calibration (isotonic),
6. construit une **Master Scale** (grille de notation),
7. produit des fichiers scorés (Train / Validation / OOS),
8. génère des **rapports HTML** :

   * rapport global Train vs Validation,
   * rapports **Vintage / Grade** (validation, OOS) avec analyse de calibration par grade et par vintage.

L’architecture est pensée pour :

* **robustesse temporelle** (splits par vintages),
* **prévention du Data Leakage**,
* **interprétabilité** (WOE, LR, master scale, tables TTC),
* **facilité d’exploitation** (Makefile + artefacts gelés).

---

## Architecture du Projet

```plaintext
.
├── Makefile                     # Orchestrateur principal (Entry Point)
├── config.yml                   # Configuration (chemins, fenêtres temporelles)
├── pyproject.toml               # Dépendances (Poetry)
├── README.md                    # Documentation
├── src/
│   ├── make_labels.py           # Génération des labels (default_24m) & vintage
│   ├── impute_and_save.py       # Imputation stricte (Fit Train / Transform All)
│   ├── fit_binning.py           # Binning monotone (Max Gini, denylist)
│   ├── train_model.py           # WOE, interactions, LR, calibration, Master Scale
│   ├── apply_model.py           # Rejeu complet de la pipeline sur OOS / custom
│   ├── generate_report.py       # Rapport HTML global (Train vs Validation)
│   ├── generate_vintage_grade_report.py  # Rapport Vintage / Grade + calibration
│   └── features/
│       └── binning.py           # Implémentation détaillée du binning max|Gini|
├── data/
│   ├── raw/                     # Données brutes
│   └── processed/
│       ├── default_labels/      # Labels générés (Train / Validation / OOS)
│       ├── imputed/             # Données imputées (Train / Validation)
│       ├── binned/              # Données binned (Train / Validation)
│       └── scored/              # Fichiers scorés (Train / Val / OOS / custom)
└── artifacts/
    ├── imputer/                 # imputer.joblib (fit sur Train)
    ├── binning_maxgini/         # bins.json (spécification du binning)
    └── model_from_binned/       # model_best.joblib, risk_buckets.json, bucket_stats.json
```

---

## Démarrage Rapide

### Prérequis

* Python 3.9+
* Poetry
* Make

### Installation

```bash
poetry install
```

### Pipeline complet (Train + Validation + Report)

```bash
make pipeline
```

Cette commande enchaîne :

1. `make labels`
2. `make impute`
3. `make binning_fit`
4. `make model_train`
5. `make report`

---

## Détail des Étapes du Pipeline

Le pipeline est orchestré par le `Makefile` et respecte strictement le flux temporel (anti-leakage).

### 1. Génération des Labels — `make labels`

Script : `src/make_labels.py`

* Construction de la variable cible `default_24m` (défaut dans les 24 mois suivant l’origination).
* Génération d’un **vintage** (typ. `YYYYQn`) pour chaque prêt.
* Split temporel typique :

  * **Train** : vintages anciens (données matures).
  * **Validation** : vintages plus récents (test out-of-time).
  * **OOS** (facultatif) : échantillon hors période de modélisation.

Les fichiers labellisés sont stockés dans `data/processed/default_labels/window=24m/`.

---

### 2. Imputation — `make impute`

Script : `src/impute_and_save.py`
Sorties principales :

* `data/processed/imputed/train.parquet`
* `data/processed/imputed/validation.parquet`
* `artifacts/imputer/imputer.joblib`

Principe :

* **Fit de l’imputer uniquement sur Train** (médianes / modes, etc.).
* Application de cet imputer sur :

  * Train,
  * Validation,
  * (plus tard) OOS via `apply_model.py`.
* Option `--fail-on-nan` pour refuser toute fuite de NaN en sortie.

---

### 3. Binning monotone (Max Gini) — `make binning_fit`

Script : `src/fit_binning.py`
Sorties principaux :

* `data/processed/binned/train.parquet`
* `data/processed/binned/validation.parquet`
* `artifacts/binning_maxgini/bins.json`

Fonctionnalités :

* Binning sur les données imputées (Train / Validation).
* Utilisation de `features.binning.run_binning_maxgini_on_df` :

  * variables numériques et catégorielles,
  * contrainte de **monotonicité** des taux de défaut par bin,
  * denylist stricte (`DENYLIST_STRICT_DEFAULT`) pour exclure :

    * dates / proxies de temps (`quarter`, `year`, `month`, `vintage`, etc.),
    * variables techniques (`__file_quarter`, IDs…),
  * taille minimale de bin (ex : `--min-bin-size-num 300`),
  * nombre max de bins (ex : `--max-bins-num 10`).

Les bornes et mappages sont sérialisés dans `bins.json` et seront réutilisés pour OOS / custom.

---

### 4. Entraînement du modèle — `make model_train`

Script : `src/train_model.py`
Entrées :

* `data/processed/binned/train.parquet`
* `data/processed/binned/validation.parquet`

Sorties :

* `artifacts/model_from_binned/model_best.joblib`

  * `woe_maps` (dictionnaires WOE par variable binned)
  * `best_lr` (régression logistique L2)
  * `model_pd` (calibrateur isotonic + LR)
  * `kept_features` (liste de variables WOE + interactions retenues)
* `artifacts/model_from_binned/risk_buckets.json`

  * bords de la master scale (edges de score TTC)
* `artifacts/model_from_binned/bucket_stats.json`
* `data/processed/scored/train_scored.parquet`
* `data/processed/scored/validation_scored.parquet`

#### 4.1 WOE + Interactions

* WOE appris sur **100% du Train binned**.
* Fonctions clés :

  * `build_woe_maps` → mappe chaque bin à un WOE, avec smoothing.
  * `apply_woe` → applique les maps WOE aux colonnes `__BIN`.
  * `add_interactions` → interactions WOE (ex : `credit_score_WOE * original_cltv_WOE`, etc.).

#### 4.2 Feature Selection

* Calcul sur un sous-échantillon interne `X_model` (split du Train) :

  * tri des features par variance,
  * suppression des variables avec corrélation > 0.85 (`corr_thr`).

#### 4.3 Modèle Score (Logistic Regression)

* `LogisticRegression` L2, réglée par `GridSearchCV` sur C,
* Cross-validation stratifiée (5 folds par défaut),
* `best_lr` = modèle final sélectionné.

#### 4.4 Score TTC

* Score TTC = transformation linéaire des log-odds :

  [
  \text{score_ttc} = \text{offset} - \text{factor} \times \text{log-odds}
  ]

  avec :

  * base : 600 points,
  * odds de base : 50,
  * PDO : 20 points.

* Calcul sur **tout le Train** et **toute la Validation**.

#### 4.5 Calibration PD (Isotonic Regression)

* `CalibratedClassifierCV(best_lr, method="isotonic", cv="prefit")`.
* `model_pd.predict_proba(X)[:, 1]` = PD calibrée.
* `pd_tr` et `pd_va` utilisés :

  * pour les métriques,
  * pour les fichiers `*_scored.parquet`,
  * pour la master scale.

#### 4.6 Master Scale (Grille de notation)

* Construction de buckets de score TTC via `create_risk_buckets` (10 par défaut) :

  * edges de score (quantiles Train),
  * calcul de PD par bucket (Train / Validation),
  * test de **monotonicité de la PD** par bucket.
* Sauvegarde :

  * `risk_buckets.json` (edges),
  * `bucket_stats.json` (statistiques par bucket).

#### 4.7 Fichiers train_scored / validation_scored

`train_model.py` produit directement :

* `data/processed/scored/train_scored.parquet`
* `data/processed/scored/validation_scored.parquet`

Avec au minimum :

* **target** : `default_24m`
* **score TTC** : `score_ttc`
* **PD calibrée** : `pd`
* **grade** (1..N buckets)
* colonnes meta importantes, ex :

  * `vintage`
  * `loan_sequence_number` (ID prêt)

Ces fichiers sont la base du reporting et des analyses vintage/grade.

---

### 5. Rapport Global Train vs Validation — `make report`

Script : `src/generate_report.py`
Entrées :

* `data/processed/scored/train_scored.parquet`
* `data/processed/scored/validation_scored.parquet`
* `artifacts/model_from_binned/model_best.joblib`

Sortie :

* `reports/model_validation_report.html`

Contenu :

* **Métriques globales** (Train / Validation) :

  * AUC, Gini, LogLoss, Brier, ECE.
* **ROC Curve** Train vs Validation.
* **Calibration Curve** Train vs Validation.
* **Score Distribution** (comparaison des distributions de `score_ttc`).
* **Master Scale Analysis** :

  * volumes par grade,
  * PD observée vs PD prédite par grade (Train & Val),
  * tables TTC par grade :

    * `pd_obs` (observée),
    * `pd_ttc` (référence TTC issue de la prédiction modèle),
* **Coefficients du modèle** :

  * barplot des coefficients,
  * intercept,
  * table détaillée.

Style : **dark mode** homogène avec le reste des rapports.

---

## Scoring OOS & Fichiers Custom

### Scoring OOS — `make oos_score`

Script : `src/apply_model.py`
Flux rejoué entièrement :

1. Imputation (`imputer.joblib`)
2. Binning (`bins.json`)
3. WOE + interactions (`woe_maps`)
4. Sélection de features (`kept_features`)
5. LR + calibration (`best_lr`, `model_pd`)
6. Score TTC + grade (`risk_buckets.json`)

Entrée :

* `data/processed/default_labels/window=24m/oos.parquet`

Sortie :

* `data/processed/scored/oos_scored.parquet`

Colonnes clés :

* `loan_sequence_number`
* `vintage` (si présent dans les données d’origine)
* `score` (score TTC appliqué à OOS)
* `pd` (PD calibrée)
* `grade` (bucket master scale)
* `default_24m` (si disponible dans OOS)

Commande :

```bash
make oos_score
```

### Scoring Custom — `make score_custom`

Permet de scorer n’importe quel fichier parquet/csv compatible avec le schéma attendu :

```bash
make score_custom CUSTOM_DATA=path/to/file.parquet CUSTOM_OUT=path/to/out.parquet
```

Le script `apply_model.py` rejoue exactement la même pipeline (impute + binning + WOE + LR + calibration + master scale).

---

## Rapports Vintage / Grade & Calibration

En plus du rapport global, un script dédié permet d’analyser la **distribution des grades par vintage** et la **calibration par grade / vintage**.

Script : `src/generate_vintage_grade_report.py`
Entrée : un fichier **scoré** (Train, Validation, OOS, Custom…) contenant :

* `vintage` (ou équivalent, configurable),
* `grade`,
* `pd` (PD modèle calibrée),
* `default_24m` (si on veut la calibration et les métriques).

### Contenu du Vintage / Grade Report

* **Titre dynamique** :
  `Vintage / Grade Report – <sample-name> (vintage_min → vintage_max)`

* **Section 0 — Global Scoring Metrics** (si `pd` + target) :

  * AUC, Gini, LogLoss, Brier, ECE, N, default rate.

* **Section 1 — Volumes par vintage & grade** :

  * tableau des **n** par (vintage, grade),
  * tableau des **% de volume** par (vintage, grade).

* **Section 2 — Distribution des grades par vintage** :

  * barplot empilé des grades (en % de volume par vintage).

* **Section 3 — PD moyenne par vintage et par grade** (si `pd` dispo) :

  * lignes de PD moyenne par grade vs vintage.

* **Section 4 — Taux de défaut par vintage** (si target dispo) :

  * courbe du default rate observé par vintage.

* **Section 5 — Calibration par vintage / grade vs PD TTC master scale** :

  Tables par vintage avec colonnes :

  * `grade`
  * `n` : nombre d’expositions dans ce couple (vintage, grade)
  * `n_defaults` : nombre de défauts observés (si cible dispo)
  * `pd_hat` : PD moyenne du modèle (moyenne de `pd` sur ce couple)
  * `pd_obs` : taux de défaut observé sur ce couple (si cible dispo)
  * `pd_ttc` : **PD TTC de référence pour ce grade**, définie comme :

    > moyenne des PD modèle `pd` par grade, sur **tout le fichier scoré**
    > (sert de référence TTC master scale pour ce sample).

  Une légende est affichée dans le HTML pour clarifier ces colonnes.

* **Commentaire de monotonie** :

  Pour chaque colonne de PD (`pd_ttc`, `pd_hat`, `pd_obs`), le script vérifie si la PD est **monotone croissante** avec le grade (agrégée sur tous les vintages) et affiche :

  * “monotonie respectée”
  * ou “**monotonie cassée**” pour les colonnes concernées.

### Vintage Report sur OOS — `make oos_vintage_report`

Chaîné à `oos_score`, il génère :

* `data/processed/scored/oos_scored.parquet`
* `reports/vintage_grade_oos.html`

Commande :

```bash
make oos_vintage_report
```

### Vintage Report sur Validation — `make val_vintage_report`

Sur le fichier `validation_scored.parquet` produit par `train_model.py` :

* Entrée : `data/processed/scored/validation_scored.parquet`
* Sortie : `reports/vintage_grade_validation.html`
* `sample-name = "Validation"`

Commande :

```bash
make val_vintage_report
```

---

## Récapitulatif des Principales Commandes Make

* **Pipeline complet (Train + Val + Report)**

  ```bash
  make pipeline
  ```

* **Étapes unitaires**

  ```bash
  make labels
  make impute
  make binning_fit
  make model_train
  make report
  ```

* **Scoring OOS + Vintage Report**

  ```bash
  make oos_score
  make oos_vintage_report
  ```

* **Vintage / Grade Report sur Validation**

  ```bash
  make val_vintage_report
  ```

* **Scoring custom**

  ```bash
  make score_custom CUSTOM_DATA=path/to/file.parquet CUSTOM_OUT=path/to/out.parquet
  ```

* **Nettoyage complet**

  ```bash
  make clean_all
  ```

---

## Mécanismes de Sécurité & Bonnes Pratiques

* **Anti-Leakage** :

  * Imputation fit uniquement sur Train.
  * Binning et WOE appris sur Train puis gelés.
  * Aucune variable temporelle explicite ou proxy (`vintage`, `quarter`, etc.) utilisée comme feature.

* **Monotonicité** :

  * Binning monotone (Gini vs default rate) via `features.binning`.
  * Master scale avec vérification de monotonie PD vs grade.
  * Rapport Vintage / Grade avec commentaire explicite si la monotonie de `pd_ttc`, `pd_hat`, `pd_obs` est cassée.

* **Interprétabilité** :

  * Modèle principal = Logistic Regression linéaire en WOE.
  * Interactions limitées et explicites.
  * Coefficients disponibles et visualisés dans le rapport HTML.

---

## Auteurs & Licence

Ce projet a été développé dans un contexte de calibration et validation de modèles PD (“bank-grade”).

* **Licence** : Propriétaire / Interne.
* Utilisation : expérimentation, backtesting, stress tests, et intégration potentielle dans des chaînes de production de scoring.
