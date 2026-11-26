import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.spatial.distance import jensenshannon


def calculate_jaccard(row):
    """Calculates Jaccard Similarity for Top 3 sets"""
    set1 = set([row['cat_old_1'], row['cat_old_2'], row['cat_old_3']])
    set2 = set([row['cat_new_1'], row['cat_new_2'], row['cat_new_3']])

    # Remove NaNs if any models returned fewer than 3 categories
    set1 = { x for x in set1 if pd.notna(x) }
    set2 = { x for x in set2 if pd.notna(x) }

    if not set1 and not set2:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def analyze_model_comparison(df):
    print("=== MODEL COMPARISON REPORT ===\n")

    # 1. TOP-1 AGREEMENT & COHEN'S KAPPA
    top1_match = (df['conf_old_1'] == df['conf_new_1']).mean()
    print(f"1. Top-1 Exact Match Rate: {top1_match:.2%}")

    kappa = cohen_kappa_score(df['cat_old_1'], df['cat_new_1'])
    print(f"2. Cohen's Kappa (Top-1): {kappa:.4f} "
          f"({'Excellent' if kappa > 0.8 else 'Good' if kappa > 0.6 else 'Fair' if kappa > 0.4 else 'Poor'})")

    # 2. TOP-3 JACCARD SIMILARITY
    df['top3_jaccard'] = df.apply(calculate_jaccard, axis=1)
    print(f"3. Mean Top-3 Jaccard Similarity: {df['top3_jaccard'].mean():.4f}")

    # 3. CONFIDENCE ANALYSIS
    mean_conf_m1 = df['conf_old_1'].mean()
    mean_conf_m2 = df['conf_new_1'].mean()
    print(f"\n4. Mean Top-1 Confidence: Model 1 = {mean_conf_m1:.4f} | Model 2 = {mean_conf_m2:.4f}")

    # 4. HIGH CONFIDENCE DISAGREEMENTS (Critical for manual review)
    # Both models are > 80% confident, but predict entirely different top categories
    high_conf_threshold = 0.80
    disagreements = df[(df['cat_old_1'] != df['cat_new_1']) &
                       (df['conf_old_1'] >= high_conf_threshold) &
                       (df['conf_new_1'] >= high_conf_threshold)]

    print(f"5. High-Confidence Disagreements (>80% cert): {len(disagreements)} rows "
          f"({len(disagreements) / len(df):.2%})")

    # 5. MACRO DISTRIBUTION SHIFT (Jensen-Shannon Divergence)
    # Count frequencies of Top 1 predictions for both models
    dist_m1 = df['cat_old_1'].value_counts(normalize=True).reindex(df['cat_old_1'].unique(), fill_value=0)
    dist_m2 = df['cat_new_1'].value_counts(normalize=True).reindex(dist_m1.index, fill_value=0)

    js_div = jensenshannon(dist_m1, dist_m2)
    print(f"\n6. Jensen-Shannon Divergence (Macro-Distribution): {js_div:.4f}")
    print("   (0.0 = identical category distributions, 1.0 = completely disjoint)")

    # 6. TOP SHIFTS (Transition Matrix Insights)
    # What are the most common changes when the models disagree?
    top_n_count = 10
    shifts = df[df['cat_old_1'] != df['cat_new_1']]
    shift_counts = shifts.groupby(['cat_old_1', 'cat_new_1']).size().reset_index(name='count')
    shift_counts = shift_counts.sort_values('count', ascending=False).head(top_n_count)

    print(f"\n7. Top {top_n_count} Systematic Category Shifts (M1 -> M2):")
    for _, row in shift_counts.iterrows():
        print(f"   - '{row['cat_old_1']}' shifted to '{row['cat_new_1']}' ({row['count']} times)")

    return disagreements



if __name__ == '__main__':
    file_path ="output2.csv"
    df = pd.read_csv(file_path)
    outliers_to_review = analyze_model_comparison(df)
    outliers_to_review.to_csv('needs_human_review.csv', index=False)