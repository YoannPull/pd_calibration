# Résumé flash — PD **TTC** vs **PIT**, recalibrage, re-fit & bins

## 1) Définitions clés

* **Modèle de scoring (moteur TTC)** : différencie le risque de manière *stable* (features structurelles, fenêtre longue).
  → Après **calibration long-terme** (ajustement d’intercept vers la moyenne de cycle), il produit une **PD TTC individuelle** (p_i^{\text{TTC}}).
* **PD TTC de grade** ( \text{PD}^{\text{TTC}}_g ) : valeur **de référence (LRA)** attribuée à chaque **grade** (cuts figés), **stable** et **versionnée**.
  Elle est **alignée** sur l’historique (moyenne de cycle observée) **et** sur la sortie TTC du modèle (via calibration).
* **PD PIT** (p_{i,t}^{\text{PIT}}) : PD **courante** (varie dans le temps) obtenue en ajustant la TTC au **contexte macro** (overlay).

## 2) TTC → PIT (overlay Merton–Vasicek)

Formule (facteur unique) :
[
p_{i,t}^{\text{PIT}}=\Phi!\left(\frac{\Phi^{-1}!\big(p_i^{\text{TTC}}\big)-\sqrt{\rho},Y_t}{\sqrt{1-\rho}}\right)
]

* (\rho) : corrélation d’actif (portefeuille/segment).
* (Y_t) : facteur macro (moyenne 0, variance 1), **estimé** depuis les défauts agrégés puis **relié** aux variables macro pour nowcast/forecast.
* **PD PIT de grade à (t)** : moyenne des (p_{i,t}^{\text{PIT}}) des comptes du grade.

## 3) Recalibrage vs Re-fit (quand et quoi changer)

* **Recalibrer** (niveau/pente, pas la structure)

  * *Quand ?* AUC/KS stables, mais écart PD prévue vs observée.
  * *Comment ?* Intercept-only → Platt (slope+intercept) → Isotonic (parcimonie).
  * *Impact ?* Les **PD PIT** (individuelles & de grade) sont **recalculées**. **PD TTC de grade (LRA)** **inchangée**.
* **Re-fit** (réestimer le modèle)

  * *Quand ?* **Discrimination baisse** (AUC ↓), concept drift.
  * *Comment ?* Fenêtre glissante récente, **CV temporelle** (forward-chaining), **retuning léger**, pondération de récence.
  * *Impact ?* Nouvelles PD individuelles → **PD PIT de grade** recalculée. **PD TTC de grade** ne change **que** si nouvelle **version** du modèle/grille.

## 4) Bins & ablation

* **Par défaut, garde les bins** (stabilité/audit). Les WOE s’ajustent à chaque fit.
* **Re-binner** seulement si : **PSI(feature)** élevé **et** chute Gini/KS de la variable **et** gain OOT prouvé.
* **Ablation “PSI-driven”** : OK **si encadrée** (décisions sur fenêtre CAL, validation OOT, ΔAUC toléré, gain PSI non trivial).
* **Stabiliser avant de dropper** : detrend par vintage, capping/winsorization cohérents, recalibration.

## 5) Splits & monitoring

* **Splits** :

  * Tuning : **CV temporelle** (pas de shuffle).
  * **OOT** final séparé (2–4 trimestres).
* **Déclencheurs** :

  * PSI(score) > 0,25 → recalib immédiate ; si AUC ↓ → re-fit.
  * ΔAUC ≥ 0,02–0,03 ou slope loin de 1 → recalib.
  * Δ base rate ≥ 20–30 % → **intercept shift**.

## 6) À retenir (governance)

* **Ce qui bouge souvent** : PD **PIT** (individuelle & de grade), via overlay macro et recalibration.
* **Ce qui reste fixe** (jusqu’à nouvelle version) : **grille de grades** (cuts) et **PD TTC de grade (LRA)**.
* **Objectif du modèle** : maximiser la vraisemblance / minimiser LogLoss (pas “minimiser le PSI”). Le **PSI** sert à **détecter** la dérive, pas à entraîner.

---

**TL;DR** :

1. Apprends un **moteur TTC** sur long historique, calibre-le à la **moyenne de cycle**.
2. Génère la **PIT** via **Merton–Vasicek** (facteur macro).
3. **Recalibre** souvent (intercept/Platt/isotonic) ; **re-fit** si AUC baisse.
4. **Garde bins & grades stables** ; re-bins/ablation seulement si diagnostic solide **et** gain OOT démontré.
5. **PD PIT de grade = recalculée** en continu ; **PD TTC de grade = ancre** (LRA) versionnée.
