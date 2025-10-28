# ============================================================
# ADD-ON : Décision recalibrage/refit + Grades + Tableaux par trimestre
# ============================================================

# --------- 0) Petites utilitaires ---------
def _safe_prob(v, eps=1e-12):
    v = np.asarray(v, float)
    return np.clip(v, eps, 1-eps)

def _safe_edges_from_quantiles(p, n_grades=10):
    """Bornes strictement croissantes en [0,1]. Grade 1 = plus risqué (haute PD)."""
    p = _safe_prob(p)
    q = np.quantile(p, np.linspace(0, 1, n_grades+1))
    # Uniquifier et forcer la stricte croissance
    q = np.array(q, dtype="float64")
    for i in range(1, len(q)):
        if not (q[i] > q[i-1]):
            q[i] = np.nextafter(q[i-1], np.inf)
    q[0] = max(0.0, min(q[0], 0.0))   # ~0
    q[-1] = min(1.0, max(q[-1], 1.0)) # ~1
    return q

def assign_grades_from_probs(p, edges):
    """
    Assigne des grades à partir des probabilités p et bornes edges (croissantes).
    Retourne un Series d'entiers dans {1..G}, où 1 = plus risqué.
    """
    p = _safe_prob(p)
    # bins: [edge[k], edge[k+1]] ; np.digitize retourne index de bin 0..G-1 (croissant en valeur)
    idx = np.digitize(p, edges[1:-1], right=True)  # 0..G-1
    G = len(edges) - 1
    # On veut 1 = plus risqué (haut de distribution)
    grades = (G - idx).astype(int)
    return pd.Series(grades, index=np.arange(len(p)), name="grade")

def prior_shift_adjust(p, base_train, base_curr, eps=1e-9):
    """Ajuste l'intercept (prior shift) pour bouger la moyenne de PD."""
    p = _safe_prob(p, eps)
    logit = np.log(p/(1-p))
    delta = np.log((base_curr+eps)/(1-base_curr+eps)) - np.log((base_train+eps)/(1-base_train+eps))
    z = logit + delta
    return 1 / (1 + np.exp(-z))

def platt_fit(p, y):
    """Platt scaling sur 1 feature : logit(p) -> LR -> renvoie une fonction calibratrice."""
    p = _safe_prob(p)
    z = np.log(p/(1-p)).reshape(-1, 1)
    lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
    lr.fit(z, y.astype(int))
    def calibrate(p_new):
        pn = _safe_prob(p_new)
        zn = np.log(pn/(1-pn)).reshape(-1, 1)
        return lr.predict_proba(zn)[:, 1]
    return calibrate, lr

# --------- 1) Décision : recalibrer vs refit ----------
def decide_recalib_or_refit(p_tr, y_tr, p_val, y_val,
                            auc_drop_thr=0.02,         # ΔAUC train→val qui déclenche refit
                            slope_thr=0.15,            # |slope-1| qui déclenche recalib
                            intercept_thr=0.20,        # |intercept| (en log-odds)
                            psi_thr=0.25):             # PSI(probas) train→val
    """
    Renvoie dict {action: 'none'|'recalibrate'|'refit', diagnostics: {...}}
    """
    auc_tr = roc_auc_score(y_tr, p_tr)
    auc_va = roc_auc_score(y_val, p_val)
    a, b = calibration_slope_intercept(y_val, p_val)  # intercept 'a', slope 'b'
    psi_scores = psi(p_tr, p_val, bins=10)

    action = "none"
    if (auc_tr - auc_va) >= auc_drop_thr:
        action = "refit"
    elif (abs(b-1.0) > slope_thr) or (abs(a) > intercept_thr) or (psi_scores > psi_thr):
        action = "recalibrate"

    return {
        "action": action,
        "diagnostics": {
            "auc_train": float(auc_tr),
            "auc_val": float(auc_va),
            "delta_auc": float(auc_tr - auc_va),
            "slope": float(b),
            "intercept": float(a),
            "psi_proba": float(psi_scores),
        }
    }

# --------- 2) Grille de grades sur TRAIN + référence PD de grade ----------
def build_grade_spec_from_train(p_train, y_train, n_grades=10):
    """
    Fige la grille (edges) sur TRAIN et calcule la PD attribuée par grade
    (référence = proportion de défaut OBSERVÉE sur le TRAIN).
    """
    edges = _safe_edges_from_quantiles(p_train, n_grades=n_grades)
    grades_tr = assign_grades_from_probs(p_train, edges)
    tab_tr = (pd.DataFrame({"grade": grades_tr, "y": y_train.astype(int), "p": p_train})
                .groupby("grade", dropna=False)
                .agg(n=("y","size"),
                     events=("y","sum"),
                     pd_obs=("y","mean"),
                     pd_pred_mean=("p","mean"))
                .reset_index()
                .sort_values("grade"))

    # bornes par grade (probabilité)
    lo = []; hi = []
    G = len(edges)-1
    for g in range(1, G+1):
        k = G - g  # rappel: grade 1 = plus risqué
        lo.append(edges[k])
        hi.append(edges[k+1])
    tab_tr["lo_prob"] = lo
    tab_tr["hi_prob"] = hi

    # référence attribuée (ici = pd_obs du train)
    pd_ref_by_grade = tab_tr.set_index("grade")["pd_obs"].to_dict()

    return {"edges": edges, "ref_table": tab_tr, "pd_ref_by_grade": pd_ref_by_grade}

# --------- 3) Tableau par trimestre/dataset ----------
def grade_table_for_dataset(p, y, edges, pd_ref_by_grade, name="DATASET"):
    """
    Construit le tableau demandé pour un dataset (ex: un trimestre):
    - grade
    - n
    - events
    - rate (observé)
    - lo_prob, hi_prob (bornes de la classe)
    - pd_ref (probabilité attribuée à la classe, fixée depuis TRAIN)
    """
    grades = assign_grades_from_probs(p, edges)
    df = pd.DataFrame({"grade": grades, "y": y.astype(int), "p": p})
    tab = (df.groupby("grade", dropna=False)
             .agg(n=("y","size"),
                  events=("y","sum"),
                  rate=("y","mean"),
                  pd_pred_mean=("p","mean"))
             .reset_index()
             .sort_values("grade"))
    # bornes
    lo, hi = [], []
    G = len(edges)-1
    for g in range(1, G+1):
        k = G - g
        lo.append(edges[k]); hi.append(edges[k+1])
    tab["lo_prob"] = lo
    tab["hi_prob"] = hi
    # référence (observée train)
    tab["pd_ref_grade"] = tab["grade"].map(pd_ref_by_grade)
    tab["dataset"] = name
    cols = ["dataset","grade","n","events","rate","lo_prob","hi_prob","pd_ref_grade","pd_pred_mean"]
    return tab[cols]

# --------- 4) Orchestration pour le bloc actuel (ex: VALIDATION) ----------
# 4.a) Décision recalibrage/refit (sur le bloc val existant)
if y_val is not None:
    decision = decide_recalib_or_refit(p_tr_curr, y_train_full, p_va_curr, y_val,
                                       auc_drop_thr=0.02, slope_thr=0.15, intercept_thr=0.20, psi_thr=0.25)
    print("\n[DECISION] action =", decision["action"], "| diagnostics =", decision["diagnostics"])

    # 4.b) Appliquer recalibrage si demandé (Platt sur VAL pour backtest) ; sinon garder p_va_curr
    p_val_used = p_va_curr.copy()
    p_tr_used  = p_tr_curr.copy()
    if decision["action"] == "recalibrate":
        platt_cal, _ = platt_fit(p_va_curr, y_val)
        p_val_used = platt_cal(p_va_curr)
        p_tr_used  = platt_cal(p_tr_curr)  # pour comparer à distribution train calibrée de la même façon

    # 4.c) (optionnel) Intercept-shift si fort décalage de base rate (production-like)
    # base_train = float(np.mean(y_train_full)); base_val = float(np.mean(y_val))
    # p_val_used = prior_shift_adjust(p_val_used, base_train, base_val)

    # 4.d) Construire la grille de grades sur TRAIN et la référence par grade (observée)
    GRADE_COUNT = 10
    grade_spec = build_grade_spec_from_train(p_train=p_tr_used, y_train=y_train_full, n_grades=GRADE_COUNT)
    print("\n[GRADES] edges (probabilities):", grade_spec["edges"])

    # 4.e) Tableau TRAIN (référence) et VAL (courant)
    grade_tab_train = grade_table_for_dataset(p=p_tr_used, y=y_train_full,
                                              edges=grade_spec["edges"],
                                              pd_ref_by_grade=grade_spec["pd_ref_by_grade"],
                                              name="TRAIN")
    grade_tab_val   = grade_table_for_dataset(p=p_val_used, y=y_val,
                                              edges=grade_spec["edges"],
                                              pd_ref_by_grade=grade_spec["pd_ref_by_grade"],
                                              name="VAL")

    # 4.f) Affichage / export
    print("\n=== TABLEAU PAR GRADE — TRAIN (référence) ===")
    print(grade_tab_train.to_string(index=False))
    print("\n=== TABLEAU PAR GRADE — VAL (période courante) ===")
    print(grade_tab_val.to_string(index=False))

    # Sauvegarde CSV si besoin :
    # grade_tab_train.to_csv("grades_train_reference.csv", index=False)
    # grade_tab_val.to_csv("grades_val_report.csv", index=False)

# --------- 5) Fonction générique pour d'autres trimestres ---------
def quarterly_grade_report(X_quarter, y_quarter, calibrator, edges, pd_ref_by_grade,
                           name, apply_prior_shift=False, base_train=None):
    """
    Produit le tableau demandé pour *n'importe quel trimestre* (X_quarter, y_quarter).
    - utilise le calibrateur passé (ex: cal_curr)
    - peut appliquer un prior-shift sur la moyenne si base_train & apply_prior_shift=True
    """
    p = calibrator.predict_proba(X_quarter)[:, 1]
    if apply_prior_shift and base_train is not None:
        base_curr = float(np.mean(y_quarter))
        p = prior_shift_adjust(p, base_train=base_train, base_curr=base_curr)
    return grade_table_for_dataset(p=p, y=y_quarter, edges=edges, pd_ref_by_grade=pd_ref_by_grade, name=name)

# Exemple d'usage ultérieur :
# base_train = float(np.mean(y_train_full))
# tab_Q1_2025 = quarterly_grade_report(X_Q1_2025, y_Q1_2025, cal_curr,
#                                      edges=grade_spec["edges"],
#                                      pd_ref_by_grade=grade_spec["pd_ref_by_grade"],
#                                      name="Q1-2025",
#                                      apply_prior_shift=True, base_train=base_train)
# print(tab_Q1_2025)
