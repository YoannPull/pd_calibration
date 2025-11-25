import joblib
import json
import sys
from pathlib import Path

def inspect():
    print("--- INSPECTION DES ARTEFACTS ---")
    
    # 1. BINS.JSON (Ce que le binning a produit)
    bins_path = Path("artifacts/binning_maxgini/bins.json")
    if bins_path.exists():
        bins = json.loads(bins_path.read_text())
        cat_vars = list(bins['cat_results'].keys())
        num_vars = list(bins['num_results'].keys())
        print(f"\n[BINS.JSON]")
        print(f"Suffixe déclaré : '{bins.get('bin_col_suffix')}'")
        print(f"Variables Cat ({len(cat_vars)}) : {cat_vars[:5]} ...")
        print(f"Variables Num ({len(num_vars)}) : {num_vars[:5]} ...")
    else:
        print("[ERR] bins.json introuvable")

    # 2. MODEL_BEST.JOBLIB (Ce que le modèle a appris)
    model_path = Path("artifacts/model_from_binned/model_best.joblib")
    if model_path.exists():
        pkg = joblib.load(model_path)
        print(f"\n[MODEL_BEST.JOBLIB]")
        
        # Kept Features (Variables retenues après sélection)
        kept = pkg.get('kept_features', [])
        print(f"Variables finales retenues ({len(kept)}) : {kept}")
        
        # WOE Maps (Variables pour lesquelles on a un mapping)
        woe_maps = pkg.get('woe_maps', {}).keys()
        print(f"Variables avec WOE Map ({len(woe_maps)}) : {list(woe_maps)}")
        
        # Vérification du mapping LR
        best_lr = pkg.get('best_lr')
        if hasattr(best_lr, 'coef_'):
            print(f"Coefficients LR : {best_lr.coef_}")
    else:
        print("[ERR] model_best.joblib introuvable")

if __name__ == "__main__":
    inspect()