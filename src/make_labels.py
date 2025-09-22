import argparse
from features.labels import build_default_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="Path to origination dataset")
    ap.add_argument("--perf", required=True, help="Path to performance dataset")
    ap.add_argument("--window", type=int, default=24, help="Default horizon in months")
    ap.add_argument("--require_full_window", action="store_true", 
                    help="Require full observation window for default calculation")
    ap.add_argument("--out", required=True, help="Path to save the processed dataset")  
    args = ap.parse_args()

    # Build dataset
    df = build_default_labels(
        path_orig=args.orig,
        path_perf=args.perf,
        window_months=args.window,
        require_full_window=args.require_full_window
    )

    # Save dataset
    df.to_parquet(args.out, index=False)  
    print(f" Wrote {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
