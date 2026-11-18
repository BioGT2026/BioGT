import pandas as pd
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories_csv", default="mondo_disease_categories.csv")
    parser.add_argument("--output_dir", default="category_splits_data")
    args = parser.parse_args()

    # Handle running from root or Model dir
    if not os.path.exists(args.categories_csv):
        if os.path.exists(os.path.join("Model", "mondo_disease_categories.csv")):
            args.categories_csv = os.path.join("Model", "mondo_disease_categories.csv")
        elif os.path.exists("mondo_disease_categories.csv"):
            args.categories_csv = "mondo_disease_categories.csv"
            
    if not os.path.exists(args.categories_csv):
        sys.stderr.write(f"Error: Could not find {args.categories_csv}\n")
        sys.exit(1)
    
    df = pd.read_csv(args.categories_csv)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if 'Category' not in df.columns or 'Disease_ID' not in df.columns:
        sys.stderr.write("Error: CSV must have 'Category' and 'Disease_ID' columns.\n")
        sys.exit(1)
        
    categories = sorted(df['Category'].dropna().unique())
    
    # Print list of sanitised_cat|filepath for bash to consume
    for cat in categories:
        safe_cat = cat.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        fname = os.path.join(args.output_dir, f"{safe_cat}.txt")
        # Save IDs
        ids = df[df['Category'] == cat]['Disease_ID'].tolist()
        
        # Only write if file doesn't exist or content changed? 
        # Determining "content changed" is hard, just overwrite.
        with open(fname, "w") as f:
            for i in ids:
                f.write(f"{i}\n")
        
        # Output absolute path
        abs_path = os.path.abspath(fname)
        print(f"{safe_cat}|{abs_path}")

if __name__ == "__main__":
    main()
