#!/usr/bin/env python3
"""
Logistic Regression Analysis for Parameter Sweep Results

This script analyzes the results from parameter_sweep_asymmetric_freeing.py
using logistic regression to understand which parameters influence P(left).

The analysis includes:
1. Univariate effects of each parameter
2. Interaction effects (e.g., beta_h * gamma_h)
3. Model comparison (with/without interactions)
4. Visualization of coefficients and predictions

Usage:
    python experiments/analyze_parameter_sweep.py outputs/parameter_sweep/results.csv
    python experiments/analyze_parameter_sweep.py outputs/parameter_sweep/results.csv --interactions
    python experiments/analyze_parameter_sweep.py outputs/parameter_sweep/results.csv --output analysis_results.txt
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Try to import statsmodels for logistic regression
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import logit
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False
    print("Warning: statsmodels not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])
    import statsmodels.api as sm
    from statsmodels.formula.api import logit
    HAVE_STATSMODELS = True

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False
    print("Warning: matplotlib not available. Skipping visualizations.")


def load_results(csv_file: str) -> pd.DataFrame:
    """
    Load results from CSV file.
    
    Args:
        csv_file: Path to results CSV
        
    Returns:
        DataFrame with results
    """
    df = pd.read_csv(csv_file)
    
    # Filter out invalid results (neither human freed)
    df_valid = df[df['left_freed_first'] != -1].copy()
    
    print(f"Loaded {len(df)} total samples")
    print(f"Valid samples (at least one human freed): {len(df_valid)}")
    print(f"Left freed first: {(df_valid['left_freed_first'] == 1).sum()}/{len(df_valid)}")
    print()
    
    return df_valid


def compute_univariate_effects(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """
    Compute univariate logistic regression for each predictor.
    
    Args:
        df: DataFrame with results
        predictors: List of predictor variable names
        
    Returns:
        DataFrame with coefficients, p-values, and odds ratios
    """
    results = []
    
    for pred in predictors:
        try:
            # Fit simple logistic regression
            formula = f"left_freed_first ~ {pred}"
            model = logit(formula, data=df).fit(disp=0)
            
            # Extract coefficient and stats
            coef = model.params[pred]
            pval = model.pvalues[pred]
            odds_ratio = np.exp(coef)
            ci_low, ci_high = np.exp(model.conf_int().loc[pred])
            
            results.append({
                'predictor': pred,
                'coefficient': coef,
                'odds_ratio': odds_ratio,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_value': pval,
                'significant': pval < 0.05
            })
        except (np.linalg.LinAlgError, ValueError) as e:
            # Handle singular matrix or other fitting errors
            results.append({
                'predictor': pred,
                'coefficient': np.nan,
                'odds_ratio': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'p_value': np.nan,
                'significant': False
            })
    
    return pd.DataFrame(results)


def fit_full_model(df: pd.DataFrame, 
                  predictors: List[str],
                  include_interactions: bool = False) -> Tuple[sm.GLM, pd.DataFrame]:
    """
    Fit full logistic regression model with all predictors.
    
    Args:
        df: DataFrame with results
        predictors: List of predictor variable names
        include_interactions: Whether to include interaction terms
        
    Returns:
        Tuple of (fitted model, summary DataFrame)
    """
    # Build formula
    if include_interactions:
        # Include some plausible interactions
        interactions = [
            "beta_h:gamma_h",  # Human planning horizon affects beta_h interpretation
            "gamma_r:gamma_h",  # Discount factors may interact
            "zeta:xi",          # Risk aversion parameters may interact
            "max_steps:beta_h"  # Horizon affects planning
        ]
        formula = "left_freed_first ~ " + " + ".join(predictors) + " + " + " + ".join(interactions)
    else:
        formula = "left_freed_first ~ " + " + ".join(predictors)
    
    print(f"Fitting model: {formula}")
    try:
        model = logit(formula, data=df).fit(disp=0)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"ERROR: Could not fit model - {e}")
        print("This usually happens when there is no variance in the outcome or predictors.")
        return None, None
    
    # Extract results
    summary = pd.DataFrame({
        'coefficient': model.params,
        'std_err': model.bse,
        'z_value': model.tvalues,
        'p_value': model.pvalues,
        'odds_ratio': np.exp(model.params),
        'significant': model.pvalues < 0.05
    })
    
    return model, summary


def print_model_summary(model, summary: pd.DataFrame, output_file=None):
    """
    Print a formatted summary of the model.
    
    Args:
        model: Fitted statsmodels model
        summary: Summary DataFrame
        output_file: Optional file to write output to
    """
    def print_or_write(text, file=None):
        print(text)
        if file:
            file.write(text + '\n')
    
    f = open(output_file, 'w') if output_file else None
    
    try:
        print_or_write("=" * 80, f)
        print_or_write("LOGISTIC REGRESSION RESULTS", f)
        print_or_write("=" * 80, f)
        print_or_write("", f)
        
        # Model fit statistics
        print_or_write("Model Fit Statistics:", f)
        print_or_write(f"  Observations: {int(model.nobs)}", f)
        print_or_write(f"  Log-Likelihood: {model.llf:.2f}", f)
        print_or_write(f"  AIC: {model.aic:.2f}", f)
        print_or_write(f"  BIC: {model.bic:.2f}", f)
        print_or_write(f"  Pseudo R-squared (McFadden): {model.prsquared:.4f}", f)
        print_or_write("", f)
        
        # Coefficients table
        print_or_write("Coefficients:", f)
        print_or_write("", f)
        
        # Format table
        header = f"{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'z':>8} {'P>|z|':>8} {'Odds Ratio':>12} {'Sig':>5}"
        print_or_write(header, f)
        print_or_write("-" * len(header), f)
        
        for var, row in summary.iterrows():
            sig_marker = "***" if row['p_value'] < 0.001 else \
                        "**" if row['p_value'] < 0.01 else \
                        "*" if row['p_value'] < 0.05 else ""
            
            line = f"{var:<20} {row['coefficient']:>10.4f} {row['std_err']:>10.4f} " \
                  f"{row['z_value']:>8.3f} {row['p_value']:>8.4f} {row['odds_ratio']:>12.4f} {sig_marker:>5}"
            print_or_write(line, f)
        
        print_or_write("", f)
        print_or_write("Significance: *** p<0.001, ** p<0.01, * p<0.05", f)
        print_or_write("", f)
        
        # Interpretation
        print_or_write("=" * 80, f)
        print_or_write("INTERPRETATION", f)
        print_or_write("=" * 80, f)
        print_or_write("", f)
        print_or_write("Odds Ratio > 1: Increasing this parameter makes it MORE likely to free left human first", f)
        print_or_write("Odds Ratio < 1: Increasing this parameter makes it LESS likely to free left human first", f)
        print_or_write("Odds Ratio ≈ 1: This parameter has little effect on the decision", f)
        print_or_write("", f)
        
        # Highlight significant effects
        sig_effects = summary[summary['significant'] & (summary.index != 'Intercept')]
        if len(sig_effects) > 0:
            print_or_write("Significant Effects (p < 0.05):", f)
            for var, row in sig_effects.iterrows():
                direction = "increases" if row['odds_ratio'] > 1 else "decreases"
                magnitude = abs(row['odds_ratio'] - 1) * 100
                print_or_write(f"  {var}: {direction} P(left) by ~{magnitude:.1f}% per unit increase", f)
        else:
            print_or_write("No significant effects found (may need more data or different parameters)", f)
        
        print_or_write("", f)
        print_or_write("=" * 80, f)
        
    finally:
        if f:
            f.close()


def create_visualizations(df: pd.DataFrame, 
                         univariate_results: pd.DataFrame,
                         predictors: List[str],
                         output_dir: str):
    """
    Create visualization plots for the analysis.
    
    Args:
        df: DataFrame with results
        univariate_results: Univariate regression results
        predictors: List of predictor names
        output_dir: Directory to save plots
    """
    if not HAVE_MATPLOTLIB:
        print("Skipping visualizations (matplotlib not available)")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Coefficient plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sig_results = univariate_results[univariate_results['significant']]
    nonsig_results = univariate_results[~univariate_results['significant']]
    
    # Plot significant effects
    if len(sig_results) > 0:
        ax.errorbar(sig_results['predictor'], 
                   sig_results['coefficient'],
                   yerr=[sig_results['coefficient'] - np.log(sig_results['ci_low']),
                         np.log(sig_results['ci_high']) - sig_results['coefficient']],
                   fmt='o', color='red', label='Significant (p<0.05)', markersize=8)
    
    # Plot non-significant effects
    if len(nonsig_results) > 0:
        ax.errorbar(nonsig_results['predictor'],
                   nonsig_results['coefficient'],
                   yerr=[nonsig_results['coefficient'] - np.log(nonsig_results['ci_low']),
                         np.log(nonsig_results['ci_high']) - nonsig_results['coefficient']],
                   fmt='o', color='gray', label='Not significant', markersize=8, alpha=0.5)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Coefficient (log odds)')
    ax.set_title('Univariate Effects on P(left)')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'coefficient_plot.png', dpi=150)
    plt.close()
    
    print(f"Saved coefficient plot to {output_path / 'coefficient_plot.png'}")
    
    # 2. Scatter plots for each predictor
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, pred in enumerate(predictors):
        ax = axes[i]
        
        # Add jitter to binary outcome for better visualization
        jitter = np.random.normal(0, 0.02, size=len(df))
        y_jittered = df['left_freed_first'] + jitter
        
        ax.scatter(df[pred], y_jittered, alpha=0.5, s=20)
        
        # Add logistic fit curve
        x_range = np.linspace(df[pred].min(), df[pred].max(), 100)
        formula = f"left_freed_first ~ {pred}"
        model = logit(formula, data=df).fit(disp=0)
        
        # Predict probabilities
        pred_df = pd.DataFrame({pred: x_range})
        y_pred = model.predict(sm.add_constant(pred_df, has_constant='add'))
        
        ax.plot(x_range, y_pred, 'r-', linewidth=2, label='Logistic fit')
        
        ax.set_xlabel(pred)
        ax.set_ylabel('P(left freed first)')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(predictors), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'scatter_plots.png', dpi=150)
    plt.close()
    
    print(f"Saved scatter plots to {output_path / 'scatter_plots.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze parameter sweep results with logistic regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('results_csv', type=str,
                       help='Path to results CSV file from parameter_sweep_asymmetric_freeing.py')
    parser.add_argument('--interactions', action='store_true',
                       help='Include interaction terms in the model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for text results (default: print to stdout)')
    parser.add_argument('--plots_dir', type=str, default='outputs/parameter_sweep/plots',
                       help='Directory for output plots (default: outputs/parameter_sweep/plots)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    df = load_results(args.results_csv)
    
    if len(df) < 10:
        print("WARNING: Very few valid samples. Results may not be reliable.")
        print("Consider running parameter sweep with more samples.")
        print()
    
    # Check for outcome variance
    n_left = (df['left_freed_first'] == 1).sum()
    n_right = (df['left_freed_first'] == 0).sum()
    if n_left == 0 or n_right == 0:
        print("ERROR: No variance in outcome variable (all samples have same outcome).")
        print(f"  Left freed first: {n_left}")
        print(f"  Right freed first: {n_right}")
        print("\nLogistic regression requires variance in the outcome.")
        print("Please run with more samples to get variance in outcomes.")
        print()
        print("Parameter summary statistics:")
        print(df[['max_steps', 'beta_h', 'gamma_h', 'gamma_r', 'zeta', 'eta', 'xi']].describe())
        return
    
    # Define predictors (exclude beta_r since it's fixed)
    predictors = ['max_steps', 'beta_h', 'gamma_h', 'gamma_r', 'zeta', 'eta', 'xi']
    
    # Univariate analysis
    print("=" * 80)
    print("UNIVARIATE ANALYSIS")
    print("=" * 80)
    print()
    univariate_results = compute_univariate_effects(df, predictors)
    print(univariate_results.to_string(index=False))
    print()
    
    # Full model
    print("=" * 80)
    print("FULL MODEL")
    print("=" * 80)
    print()
    model, summary = fit_full_model(df, predictors, include_interactions=args.interactions)
    if model is None:
        print("Skipping full model summary due to fitting errors.")
    else:
        print_model_summary(model, summary, output_file=args.output)
    
    if args.output:
        print(f"\nDetailed results saved to: {args.output}")
    
    # Create visualizations
    if HAVE_MATPLOTLIB:
        print("\nCreating visualizations...")
        create_visualizations(df, univariate_results, predictors, args.plots_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
