# Simulation de dérive (drift) pour évaluer des intervalles de calibration binomiaux

## 1) Objectif

Évaluer **rigoureusement** la performance d’intervalles de confiance (IC) binomiaux (Jeffreys, Clopper–Pearson, Wilson, Normal, HDI) **sous dérive temporelle des PD** et **variabilité des tailles d’échantillon**, en mimant un usage séquentiel (validation mensuelle par grade).

On mesure à la fois :

* **Couverture** vs niveau nominal (1-\alpha).
* **Largeur (efficience)** des IC.
* **Asymétrie des rejets** (surdétection au-dessus/au-dessous).
* **ARL** (*Average Run Length*, longueur moyenne sans faux signal en régime calibré).
* **EDD** (*Expected Detection Delay*, délai moyen de détection après une dérive).

---

## 2) Cadre probabiliste formel

### 2.1 Processus « vrai »

* Indices de temps et de segments (grades) : (t=1,\dots,T), (k=1,\dots,K).
* Facteur macro latent (AR(1)) :
  [
  Z_t = \varphi Z_{t-1} + \sigma_\varepsilon ,\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,1),\ | \varphi |<1, \ Z_0\sim \mathcal N!\left(0,\frac{\sigma_\varepsilon^2}{1-\varphi^2}\right).
  ]
* **Dérive** lente (au choix) :

  * *Rampe (drift linéaire)* : (\delta_t = \kappa,(t-\tau)*+) avec ((x)*+=\max(x,0)).
  * *Marche aléatoire biaisée* : (\delta_t = \delta_{t-1} + \mu + \eta_t,\ \eta_t\sim\mathcal N(0,\sigma_\eta^2)).
  * *Rupture* : (\delta_t = 0) si (t\le \tau), (\delta_t=\Delta) sinon.
* **Vraie probabilité de défaut par grade** :
  [
  p_{k,t} = \sigma!\big(\alpha_k + \beta_k Z_t + \delta_t\big),\qquad \sigma(x)=\frac{1}{1+e^{-x}}.
  ]
* **Taille d’échantillon** (entrées mensuelles ou encours éligible) :
  [
  n_{k,t}\sim \mathrm{Poisson}!\big(\lambda_k,e^{\gamma Z_t}\big)\quad\text{(ou fixe si souhaité)}.
  ]
* **Défauts réalisés** :
  [
  d_{k,t}\sim \mathrm{Binomial}!\big(n_{k,t},, p_{k,t}\big).
  ]

### 2.2 « Modèle sous test » (figé puis OOT)

* Fenêtre **apprentissage** (t=1{:}T_0). On estime (\hat\alpha_k,\hat\beta_k) en supposant **pas de dérive** ((\delta_t\equiv 0) sur (1{:}T_0)).
* Fenêtre **hors-temps (OOT)** (t=T_0{+}1{:}T). Les prédictions (PD de modèle) sont
  [
  \widehat{PD}_{k,t}=\sigma!\big(\hat\alpha_k+\hat\beta_k Z_t\big),
  ]
  qui **n’absorbent pas** la dérive (\delta_t) (dégradation volontairement simulée).

---

## 3) Intervalles de confiance testés (à partir de (n_{k,t}, d_{k,t}))

Soit (\hat p_{k,t}=d_{k,t}/n_{k,t}), (z_\alpha) le quantile normal.

* **Jeffreys** (crédible Beta(1/2,1/2)) :
  (L = \mathrm{qBeta}(\alpha/2;\ d+0{.}5,\ n-d+0{.}5),\ \ U = \mathrm{qBeta}(1-\alpha/2;\ d+0{.}5,\ n-d+0{.}5)).
* **Clopper–Pearson** :
  (L = \mathrm{qBeta}(\alpha/2;\ d,\ n-d+1)) (si (d>0), sinon (0));
  (U = \mathrm{qBeta}(1-\alpha/2;\ d+1,\ n-d)) (si (d<n), sinon (1)).
* **Wilson** :
  [
  c=\frac{\hat p+\frac{z_\alpha^2}{2n}}{1+\frac{z_\alpha^2}{n}},\quad
  h=\frac{z_\alpha}{1+\frac{z_\alpha^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z_\alpha^2}{4n^2}},\quad
  I=[c-h,,c+h]\cap[0,1].
  ]
* **Normal (Wald)** : (I=\big[\hat p\pm z_\alpha\sqrt{\tfrac{\hat p(1-\hat p)}{n}}\big]\cap[0,1]).
* **HDI (Beta)** : ([L,U]) tel que (\int_L^U \mathrm{Beta}(x; d+0{.}5,n-d+0{.}5),dx=1-\alpha) **et** (f(L)=f(U)) (résoudre numériquement).

---

## 4) Indicateurs d’évaluation

Pour chaque ((k,t)) et méthode (R) :

* **Couverture locale** : (H_{k,t}^{(R)}=\mathbf 1{\widehat{PD}*{k,t}\in I*{k,t}^{(R)}}).
* **Largeur relative** : (W_{k,t}^{(R)}=\dfrac{U_{k,t}^{(R)}-L_{k,t}^{(R)}}{\max(\hat p_{k,t},,\varepsilon)}) (petit (\varepsilon) p.ex. (10^{-5})).
* **Signe du rejet** (si hors IC) : (S_{k,t}^{(R)}=\mathrm{sign}\big(\widehat{PD}*{k,t}-\hat p*{k,t}\big)).

Agrégats sur la fenêtre OOT et/ou par régime de PD (p.ex. (p<0{.}3%), (0{.}3{-}1%), (>!1%)) :

* (\overline{H}^{(R)}) (et écart à (1-\alpha)), (\overline{W}^{(R)}), (\Pr(S=+1)-\Pr(S=-1)).
* **ARL(_0^{(R)})** : longueur moyenne (en mois) avant le **premier** faux rejet en régime sans dérive ((\delta_t\equiv 0)).
* **EDD(_\Delta^{(R)})** : délai moyen (en mois) entre (t=\tau) (début de dérive) et le **premier** rejet après (\tau).

On peut aussi agréger par mois sur les (K) grades : (H_t^{(R)}=\mathbf 1{\exists k:\ \widehat{PD}*{k,t}\notin I*{k,t}^{(R)}}) pour approcher un FWER séquentiel.

---

## 5) Algorithme (pseudo-code)

1. **Fixer** paramètres : (K,T,T_0,\varphi,\sigma_\varepsilon,\lambda_k,\gamma,\alpha_k,\beta_k) et un scénario (\delta_t) (rampe (\kappa,\tau) ou rupture (\Delta,\tau)).
2. **Simuler** (Z_{1:T}) (AR(1)) et (\delta_{1:T}).
3. **Construire** (p_{k,t}=\sigma(\alpha_k+\beta_k Z_t+\delta_t)) et (n_{k,t}\sim\mathrm{Poisson}(\lambda_k e^{\gamma Z_t})).
4. **Tirer** (d_{k,t}\sim\mathrm{Binomial}(n_{k,t},p_{k,t})).
5. **Apprentissage (1{:}T_0)** : estimer (\hat\alpha_k,\hat\beta_k) en supposant (\delta_t=0).
6. **Prédire OOT** : (\widehat{PD}_{k,t}=\sigma(\hat\alpha_k+\hat\beta_k Z_t)) pour (t>T_0).
7. **Construire IC** (I_{k,t}^{(R)}) à partir de (n_{k,t},d_{k,t}) pour chaque méthode (R).
8. **Calculer** (H_{k,t}^{(R)},W_{k,t}^{(R)},S_{k,t}^{(R)}).
9. **Agr éger** : moyens, écarts à (1-\alpha), ARL(*0) (en répétant la simul avec (\delta\equiv 0)), EDD(*\Delta) (en répétant avec dérive).
10. **Répéter** l’expérience (M) fois et rapporter moyennes et intervalles de Monte-Carlo.

> Valeurs de départ utiles : (K\in{5,10}), (T_0=60), (T=120), (\varphi\in[0.6,0.9]), (\sigma_\varepsilon\in[0.5,1]), (\kappa\in[0.02,0.06]) (rampe douce), (\Delta\in[0.3,0.6]) en logit (≈ +35–80 % de PD).

---

## 6) « Heatmap ((n,p)) » : qu’est-ce que c’est et comment la construire ?

**Idée.** Pour juger la **qualité intrinsèque** d’un IC indépendamment du temps, on balaye un **grillage** de tailles d’échantillon (n) et de vraies PD (p), on simule (d\sim\mathrm{Bin}(n,p)), et on estime la **couverture** ou la **largeur relative**. On visualise ensuite le résultat comme une **carte de chaleur** (matrice colorée) dont :

* **axe (x)** : valeurs de (n) (p.ex. (50,100,200,500,2{,}000,10{,}000)),
* **axe (y)** : valeurs de (p) **sur échelle log** (p.ex. (10^{-4},5\cdot10^{-4},10^{-3},2\cdot10^{-3},5\cdot10^{-3},10^{-2},\dots)).
* **couleur** : métrique choisie (p.ex. (|\text{couverture}- (1-\alpha)|) ou (\mathbb E[(U-L)/p])).

**Procédure.**

1. Choisir des grilles (\mathcal N={n_j}*{j=1}^J), (\mathcal P={p_i}*{i=1}^I).
2. Pour chaque paire ((p_i,n_j)), répéter (R) fois : tirer (d\sim\mathrm{Bin}(n_j,p_i)), construire l’IC (I^{(R)}), incrémenter un compteur si (p_i\in I^{(R)}), accumuler (U-L).
3. Estimer (\widehat{\text{cov}}(p_i,n_j)=\frac{#{p_i\in I}}{R}) et (\widehat{\text{width}}(p_i,n_j)=\frac{1}{R}\sum (U-L)).
4. Tracer la **matrice** ([\widehat{\text{cov}}(p_i,n_j)]_{i,j}) (ou la largeur) en **heatmap**.
5. (Option) Calculer un **score intégré** avec des poids (w(p,n)) reflétant vos cas d’usage :
   [
   \mathrm{Score}*R=\sum*{i,j} w_{ij},\big|\widehat{\text{cov}}_R(p_i,n_j)-(1-\alpha)\big|.
   ]

Cette vue met en évidence où chaque méthode **sous-couvre** (couleur « chaude » pour un écart fort) — typiquement en **faibles (p)** et **petits (n)** pour Wald/Wilson — et où elle **sur-couvre/élargit** (CP en très petites PD).

---

## 7) Ce que vous rapportez dans le papier

* **Cartes ((n,p))** de couverture et de largeur relative pour chaque méthode (intrinsèque).
* **Séries temporelles** de (H_t^{(R)}), (\overline{W}_t^{(R)}) et distributions d’**ARL/EDD** sous les scénarios de dérive (usage réel).
* Un tableau récapitulatif : (\overline{H}^{(R)}), (\overline{W}^{(R)}), ARL(*0^{(R)}), EDD(*\Delta^{(R)}), plus l’**asymétrie des rejets**.

---

### Remarque finale

Oui, **tout se formalise** proprement : le DGP ci-dessus définit (p_{k,t}) et (n_{k,t}), les IC sont fonctions mesurables de ((n_{k,t},d_{k,t})), et **ARL/EDD** sont des **temps d’arrêt** (hitting times) d’un processus ({H_{k,t}^{(R)}}), estimés par Monte-Carlo. Ainsi, tu couvres à la fois la **qualité mathématique** des IC (heatmaps ((n,p))) et leur **comportement séquentiel** sous dérive, exactement ce que le reviewer demande.
