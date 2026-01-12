# Guide complet des **transformations de variables** pour le *scoring de crédit*

> **Fichier** : `credit_scoring_transformations.md`
> **But** : fournir une recette précise et reproductible des étapes de préparation de données et d’encodage des variables pour (i) **régression logistique** et (ii) **modèles plus complexes** (GBDT/XGBoost, Random Forest, SVM, deep learning).
> **Public** : data scientists crédit / risque de défaut (PD).



## 0) Principes généraux & garde‑fous

1. **Point d’observation (PO)** : figer la date/instant où chaque observation est “connue” (ex. date d’octroi). Toute variable doit être **mesurable au PO**.
2. **Split temporel** : `train` (t₀→t₁), `valid` (t₁→t₂), `test/OOT` (t₂→t₃). Éviter tout mélange temporel.
3. **Aucune info du futur** : pas d’agrégats calculés en utilisant `valid/test` (leakage). Binning/encodages/impacts **appris uniquement sur train**, puis **appliqués**.
4. **Pipelines** : implémenter **de bout en bout** (fit/transform) pour garantir la reproductibilité.
5. **Traçabilité** : versionner données, code, hyperparamètres, seeds, et stockage des **binning maps** / **WOE tables**.

---

## 1) Découpage des données & cibles

* **Définition de la cible (Default)** : horizon H (ex. 12 mois), fenêtre de performance (ex. 12–18 mois), exclusion des dossiers sans recul suffisant.
* **Déduplication / agrégation** : un enregistrement client/contrat par PO.
* **Filtres** : retirer outliers grossiers techniques (valeurs absurdes), variables quasi constantes (>99.9% même valeur), variables **post‑événement** (ex. incidents après PO).

---

## 2) Contrôles qualité & nettoyage

* **Types** : forcer numérique/texte/date ; harmoniser unités (€, k€ ; jours/mois).
* **Valeurs manquantes** : quantifier par variable & par segment.
* **Valeurs aberrantes** : détecter par règles métier + statistiques robustes (IQR, MAD).
* **Catégories rares** : repérer modalités à faible fréquence (p.ex. <0.5–2%).

---

## 3) Stratégies de traitement (panorama)

### 3.1 Numériques

1. **Imputation** :

   * Simple : médiane (robuste) + **indicateur de manquants** (flag 0/1) si >1–2% NA.
   * Avancé (logistique/SVM/NN) : **imputation robuste** (médiane/quantile) ; *éviter* KNN si gros volume.
   * Arbres/GBDT : souvent robustes aux valeurs extrêmes **mais exigent une imputation** si l’implémentation ne gère pas les NA nativement (scikit-learn RF/XG sans NA → imputer). LightGBM gère les NA.

2. **Traitement des extrêmes** :

   * **Capping/Winsorizing** aux quantiles (p.ex. 0.5% / 99.5%).
   * **Transformations** : `log1p`, **Yeo–Johnson** pour symétriser (utile logistique/SVM/NN).

3. **Discrétisation (binning)** — *particulièrement utile pour la logistique & scorecards* :

   * **Non supervisée** : intervalles égaux, effectifs égaux (quantiles), k‑means binning.
   * **Supervisée** :

     * **Arbre de décision** (profondeur limitée, min bin size ≥5% ou ≥ n\_min).
     * **MDLP / CHAID** (si disponible).
     * **Binning monotone** (fusion séquentielle pour assurer une **tendance monotone du taux de défaut** par bin).
   * **Règles pratiques** : 4–10 bins ; chaque bin ≥ 1–5% du sample ; éviter bins avec <\~50 défauts ; préserver lisibilité métier.

4. **WOE/IV (Weight of Evidence / Information Value)** — *classique crédit* :

   * **WOE(bin)** = `ln( (Good_bin / Good_total) / (Bad_bin / Bad_total) )`.
   * **IV** = `Σ_bins ( (Good%_bin − Bad%_bin) × WOE_bin )`.
   * **Procédure** : (i) pré‑binning, (ii) fusion → **monotonicité** du *bad rate*, (iii) calcul WOE/IV sur **train**, (iv) remplacer X par **WOE(X)**, (v) re‑contrôle monotonicité coefficients de la logistique.
   * **Gestion NA/spéciaux** : bin dédié (« missing/special »).

5. **Standardisation** (utile pour logistique/SVM/NN) : `RobustScaler` (mediane/IQR) recommandé ; `StandardScaler` si distribution \~gaussienne.

### 3.2 Catégorielles

1. **Regroupement des modalités rares** : seuil sur fréquence (ex. <1%) ou cumulé (95–99%).
2. **Ordre métier** : si catégorie ordinale (ex. *ancienneté classe A\<B\<C*), utiliser **OrdinalEncoder** conforme.
3. **One‑Hot** (OHE) : pour faible cardinalité (<\~20–50).
4. **Target / Mean Encoding** : pour forte cardinalité → **impératif : CV imbriquée** (K‑fold out‑of‑fold) + bruitage pour éviter leakage.
5. **WOE catégorielle** : même logique que numérique (bins = modalités/fusions).
6. **Hashing trick** : si cardinalité énorme (codes marchands), à utiliser prudemment.

### 3.3 Dates & temporelles

* Construire **âges** (ex. âge du compte), **récences**, **durées** (jours depuis…) ; **saisonnalité** (mois, jour-semaine) si pertinent.
* **Jamais** utiliser des infos **postérieures au PO**.

### 3.4 Texte court / adresses / libellés

* Mappages métier (catégoriser), **listes noires** (ex. certains codes), ou *bag‑of‑words* simple **appris sur train** uniquement.

---

## 4) Déséquilibre de classes (PD faible)

* **Class weights** dans la perte (logistique, SVM, GBDT, NN).
* **Sur‑échantillonnage** : **SMOTE/SMOTE‑NC** (appliquer **dans CV**, jamais avant le split).
* **Sous‑échantillonnage** contrôlé sur la majorité.
* **Seuil de décision** optimisé par **coûts** (EL = PD×LGD×EAD) ou par métriques (Fβ, PR‑AUC max, KS, etc.).

---

## 5) Sélection de variables

1. **Filtres** : IV (≥0.02 faible, ≥0.1 bon), corrélations, *mutual information*, tests univariés (χ², ANOVA).
2. **Embeddeds** : L1/L2 (logistique), *gain* GBDT, permutation importance.
3. **Wrappers** : RFE, *sequential forward/backward*.
4. **Colinéarité** : VIF (<5–10), supprimer doublons/combinaisons quasi colinéaires.
5. **Stabilité** : vérifier dérive **PSI** entre train et OOT (PSI < 0.1 idéal).
6. **Parcimonie** : préférer peu de variables → robustesse & interprétabilité.

---

## 6) Interactions & non‑linéarités

* **Logistique** :

  * ajouter **interactions** métier (produits, ratios, min/max) ;
  * **splines** (p‑splines, cubic splines), **Yeo–Johnson**, **binnings** (via WOE) pour capturer non‑linéarités.
* **GBDT/RF** : capturent naturellement non‑linéarités → interactions manuelles souvent inutiles.

---

## 7) Calibration & score final

1. **Calibration** (post‑modèle) :

   * **Platt (logistique)**, **Isotonic**, **Beta calibration**.
   * Apprendre la calibration sur **valid** (ou via CV), appliquer sur test.
2. **Traduction en **score** (PDO)** — pour logistique/scorecards :

   * Objectif : `Score = Offset + Factor × ln(odds)`.
   * Avec **PDO** = points pour **doublement des odds** → `Factor = PDO / ln(2)`.
   * Pour un **base score** `S₀` à **base odds** `O₀`: `Offset = S₀ − Factor × ln(O₀)`.
3. **Cartes de points (scorecards)** : points par bin = `round(Factor × WOE_bin × β_var)`, sommation + Offset.

---

## 8) Pipelines recommandés (par famille de modèles)

### 8.1 Régression logistique (modèle de référence réglementaire)

**Objectif** : robustesse, monotonicité, interprétabilité.

**Pipeline typique** :

1. Split temporel → `train/valid/test(OOT)`.

2. Pré‑traitements **sur train** :

   * Imputation robuste + indicateur NA.
   * **Binning supervisé monotone** (4–8 bins).
   * **WOE** pour chaque variable (incluant bin NA/spéciaux).
   * Filtre IV, VIF, parcimonie.

3. Modèle :

   * Logistique **L2** (ou **L1** pour sélection).
   * Contrôler signes attendus (monotonicité métier).

4. Calibration : isotonic ou Platt (appris sur `valid`).

5. Score : appliquer mapping PDO.

**À éviter** : encoder en OHE des variables à très forte cardinalité ; appliquer SMOTE avant split ; apprendre WOE sur train+valid ; imputer avec moyenne globale incluant test.

### 8.2 Arbres & Gradient Boosting (XGBoost / LightGBM / CatBoost / Random Forest)

* **Pré‑traitements minimaux** :

  * **Aucune standardisation nécessaire**.
  * **Imputation** : requise pour scikit RF/XGB (NA non gérés nativement), optionnelle/automatique pour LightGBM/CatBoost.
  * **Catégorielles** :

    * **CatBoost** : gère nativement (encodage ordonné).
    * **LightGBM** : `categorical_feature` + codes entiers ; ou OHE si faible cardinalité.
    * **XGBoost** : OHE ou **target encoding** (avec CV imbriquée).
* **Régulation & stabilité** : profondeur limitée, `min_child_samples`, `subsample`, `colsample_bytree`.
* **Monotonicité** : contraintes monotones possibles (XGB/LGB) pour variables clefs.
* **Poids de classe** : `scale_pos_weight` (≈ `neg/pos`) ou class weights.
* **Calibration** souvent nécessaire (Platt/Isotonic).
* **Feature selection** rarement indispensable ; utiliser importance/permutation + stabilité.

### 8.3 SVM

* **Standardisation obligatoire** (z‑score ou robust).
* OHE pour catégorielles (ou target encoding avec CV imbriquée).
* Choix du noyau : linéaire (grand N, forte dim), RBF (non‑linéaire).
* **Probabilités** : activer `probability=True` (Platt interne) ou recalibrer ex‑post.
* Coût computationnel élevé → échantillons/approx. (LinearSVM).

### 8.4 Réseaux de neurones (tabulaires)

* **Normalisation** des numériques (z‑score/robust).
* **Embeddings** pour catégorielles (entités à forte cardinalité).
* Régularisation : dropout, L2, early‑stopping.
* **Class weights** ou focal loss.
* Calibration ex‑post recommandée.
* Exiger plus de données et un fort contrôle de la dérive.

---

## 9) Checklists opérationnelles

**Avant le fit**

* [ ] PO défini, data leak check passé.
* [ ] Split temporel + OOT.
* [ ] NA, outliers, rares mappés.
* [ ] Binnings figés (sauvegardés).
* [ ] Encodages appris **uniquement** sur train.

**Après le fit**

* [ ] Discrimination : AUC/KS/PR‑AUC ok.
* [ ] **Calibration** : Brier, courbes de calibration, ECE.
* [ ] Stabilité : PSI (train vs OOT), drift score.
* [ ] Interprétabilité : signes, monotonicité, SHAP (arbres).
* [ ] Documentation & versioning.

---

## 10) Exemples **Python** (sklearn‑style)

### 10.1 Logistique avec WOE + calibration isotone

```python
# pip install pandas numpy scikit-learn category-encoders optbinning
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from optbinning import OptimalBinning

X_train, y_train = ...  # DataFrame / Series
X_valid, y_valid = ...
X_test,  y_test  = ...

woe_maps = {}
Xtr_woe = pd.DataFrame(index=X_train.index)
Xva_woe = pd.DataFrame(index=X_valid.index)
Xte_woe = pd.DataFrame(index=X_test.index)

for col in X_train.columns:
    if np.issubdtype(X_train[col].dtype, np.number):
        optb = OptimalBinning(name=col, dtype="numerical", solver="cp",
                              monotonic_trend="ascending", min_n_bins=3, max_n_bins=8)
    else:
        optb = OptimalBinning(name=col, dtype="categorical")
    optb.fit(X_train[col].values, y_train.values)
    Xtr_woe[col] = optb.transform(X_train[col].values, metric="woe")
    Xva_woe[col] = optb.transform(X_valid[col].values, metric="woe")
    Xte_woe[col] = optb.transform(X_test[col].values,  metric="woe")
    woe_maps[col] = optb.binning_table.build()  # à sérialiser

logit = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=200)
calibrated = CalibratedClassifierCV(logit, method="isotonic", cv="prefit")

logit.fit(Xtr_woe, y_train)
calibrated.fit(Xva_woe, y_valid)

proba = calibrated.predict_proba(Xte_woe)[:,1]
print("AUC:", roc_auc_score(y_test, proba))
print("Brier:", brier_score_loss(y_test, proba))
```

### 10.2 XGBoost (catégorielles gérées par OHE) + calibration Platt

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

num_cols = X_train.select_dtypes(include=[np.number]).columns
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

preprocess = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

xgb = XGBClassifier(
    n_estimators=600, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    scale_pos_weight=(y_train.shape[0]-y_train.sum())/y_train.sum()
)

pipe = Pipeline([("prep", preprocess), ("clf", xgb)])
pipe.fit(X_train, y_train, clf__eval_set=[(X_valid, y_valid)], clf__verbose=False)

platt = CalibratedClassifierCV(pipe, method="sigmoid")
platt.fit(X_valid, y_valid)

proba = platt.predict_proba(X_test)[:,1]
```

---

## 11) Annexes (formules & heuristiques)

### 11.1 WOE / IV

* `WOE_i = ln( (g_i / G) / (b_i / B) )` où `g_i` (# bons) & `b_i` (# défauts) dans le bin *i* ; `G` & `B` totaux.
* `IV = Σ_i ( (g_i/G − b_i/B) × WOE_i )`.

**Règles IV (indicatives)** : <0.02 = faible, 0.02–0.1 = utile, 0.1–0.3 = fort, >0.3 = très fort (attention overfit).

### 11.2 PSI (Population Stability Index)

`PSI = Σ_bins ( (p_i − q_i) × ln(p_i/q_i) )` où `p_i` proportion en train, `q_i` en OOT.
Seuils indicatifs : <0.1 stable ; 0.1–0.25 dérive modérée ; >0.25 dérive forte.

### 11.3 Heuristiques de binning

* Min défauts/bin ≥ 50 si possible ; min effectif/bin ≥ 1–5%.
* Bad rate **monotone** vs variable *risque* ; fusionner bins adjacents sinon.
* NA : bin séparé.
* Max \~10 bins pour lisibilité.

### 11.4 Cartographie *PDO* → score

* `Factor = PDO / ln(2)` ; `Offset = S₀ − Factor × ln(O₀)` ; `Score = Offset + Factor × ln(odds)` avec `odds = PD/(1−PD)`.

---

## 12) Résumé express (logistique)

1. Split temporel + OOT → 2) Imputation + flags NA → 3) Binning supervisé monotone → 4) WOE → 5) Filtre IV/VIF → 6) Logit L2/L1 → 7) Calibration → 8) PDO scorecard → 9) Contrôles (AUC, Brier, calibration curve, PSI, dérive) → 10) Documentation & gel des mappings.

---

## 13) Pièges fréquents (à éviter absolument)

* Binning/WOE appris sur train+valid ou sur l’ensemble → **leakage**.
* **SMOTE** appliqué avant le split ou en dehors de la CV.
* Imputation globale utilisant aussi `valid/test`.
* **Target encoding** sans schéma *out‑of‑fold* + bruitage.
* Variables post‑octroi ou post‑incident dans le modèle.
* Trop de variables / instabilité (pas de contrôle IV/VIF/PSI).
* Calibration faite sur test (doit être apprise sur valid ou via CV).

---

**Fin du document.**
