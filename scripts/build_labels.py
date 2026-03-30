import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def compute_cognitive_load_labels(
    questionnaire_path: str,
    percentile_low: float = 27,
    percentile_high: float = 73
) -> pd.DataFrame:

    df = pd.read_excel(questionnaire_path)
    df["participant_id"] = df["participant_id"].astype(str)
    icl_items = ['Q1', 'Q2', 'Q3', 'Q4']
    ecl_items = ['Q5', 'Q6', 'Q7', 'Q8']
    gcl_items = ['Q9R', 'Q10R', 'Q11R', 'Q12R']

    df['ICL_score'] = df[icl_items].sum(axis=1)
    df['ECL_score'] = df[ecl_items].sum(axis=1)
    df['GCL_score'] = df[gcl_items].sum(axis=1)
    icl_q27 = np.percentile(df['ICL_score'], percentile_low)
    icl_q73 = np.percentile(df['ICL_score'], percentile_high)
    ecl_q27 = np.percentile(df['ECL_score'], percentile_low)
    ecl_q73 = np.percentile(df['ECL_score'], percentile_high)
    gcl_q27 = np.percentile(df['GCL_score'], percentile_low)
    gcl_q73 = np.percentile(df['GCL_score'], percentile_high)
    def score_to_label(score, q27, q73):
        if score < q27:
            return 0
        elif score > q73:
            return 2
        else:
            return 1
    df['ICL_label'] = df['ICL_score'].apply(lambda x: score_to_label(x, icl_q27, icl_q73))
    df['ECL_label'] = df['ECL_score'].apply(lambda x: score_to_label(x, ecl_q27, ecl_q73))
    df['GCL_label'] = df['GCL_score'].apply(lambda x: score_to_label(x, gcl_q27, gcl_q73))
    result_df = df[['participant_id', 'ICL_label', 'ECL_label', 'GCL_label']].copy()
    return result_df

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cognitive load labels (ICL, ECL, GCL) from questionnaire data."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the questionnaire Excel file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="cognitive_load_labels.csv",
        help="Path to save the output labels CSV file."
    )
    parser.add_argument(
        "--percentile_low",
        type=float,
        default=27,
        help="Lower percentile threshold."
    )
    parser.add_argument(
        "--percentile_high",
        type=float,
        default=73,
        help="Upper percentile threshold."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    labels = compute_cognitive_load_labels(
        questionnaire_path=str(input_path),
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved labels to: {output_path}")

if __name__ == "__main__":
    main()