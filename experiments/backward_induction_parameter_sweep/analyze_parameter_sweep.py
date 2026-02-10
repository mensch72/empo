#!/usr/bin/env python3
"""
Generalized Linear Model Analysis for Parameter Sweep Results

This script analyzes the results from parameter_sweep_asymmetric_freeing.py
using GLM with logit link to understand which parameters influence P(left).

Since P(left) is a probability in [0,1], we use a quasi-binomial GLM with
logit link function (equivalent to beta regression for proportion data).

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

# Try to import statsmodels for regression
try:
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.families.links import Logit
    # Suppress BIC deprecation warning by using LLF-based formula
    SET_USE_BIC_LLF(True)
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False
    print("Warning: statsmodels not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.families.links import Logit
    SET_USE_BIC_LLF(True)
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
    
    # Verify all seeds are unique (detects accidental duplicate runs)
    if 'seed' in df.columns:
        n_unique_seeds = df['seed'].nunique()
        n_total = len(df)
        if n_unique_seeds < n_total:
            duplicate_seeds = df[df['seed'].duplicated(keep=False)]['seed'].unique()
            print(f"WARNING: Found {n_total - n_unique_seeds} duplicate seeds!")
            print(f"         Duplicate values: {list(duplicate_seeds)[:10]}{'...' if len(duplicate_seeds) > 10 else ''}")
            print(f"         This may indicate parallel tasks with overlapping seeds.")
        else:
            print(f"Seed check: All {n_unique_seeds} seeds are unique.")
    else:
        print("Warning: No 'seed' column found in CSV (older format).")
    
    # Filter out invalid results (p_left is None/NaN)
    df_valid = df[df['p_left'].notna()].copy()
    
    print(f"Loaded {len(df)} total samples")
    print(f"Valid samples: {len(df_valid)}")
    if len(df_valid) > 0:
        print(f"P(left) range: [{df_valid['p_left'].min():.4f}, {df_valid['p_left'].max():.4f}]")
        print(f"P(left) mean: {df_valid['p_left'].mean():.4f}")
    print()
    
    return df_valid


def compute_univariate_effects(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """
    Compute univariate GLM (logit link) for each predictor.
    
    Args:
        df: DataFrame with results
        predictors: List of predictor variable names
        
    Returns:
        DataFrame with coefficients, p-values, and odds ratios
    """
    results = []
    
    # Clamp p_left to avoid exact 0 or 1 (causes GLM issues)
    y = df['p_left'].clip(0.001, 0.999)
    
    for pred in predictors:
        try:
            # Fit GLM with binomial family and logit link
            X = sm.add_constant(df[pred])
            model = GLM(y, X, family=Binomial(link=Logit())).fit()
            
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
    Fit full GLM (logit link) with all predictors.
    
    Args:
        df: DataFrame with results
        predictors: List of predictor variable names
        include_interactions: Whether to include interaction terms
        
    Returns:
        Tuple of (fitted model, summary DataFrame)
    """
    # Clamp p_left to avoid exact 0 or 1
    y = df['p_left'].clip(0.001, 0.999)
    
    # Build design matrix
    X = df[predictors].copy()
    
    if include_interactions:
        # Add interaction terms
        X['beta_h_x_gamma_h'] = df['beta_h'] * df['gamma_h']
        X['gamma_r_x_gamma_h'] = df['gamma_r'] * df['gamma_h']
        X['zeta_x_xi'] = df['zeta'] * df['xi']
        X['max_steps_x_beta_h'] = df['max_steps'] * df['beta_h']
    
    X = sm.add_constant(X)
    
    print(f"Fitting GLM with predictors: {list(X.columns)}")
    try:
        model = GLM(y, X, family=Binomial(link=Logit())).fit()
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
        print_or_write("GLM RESULTS (Binomial family, logit link)", f)
        print_or_write("=" * 80, f)
        print_or_write("", f)
        
        # Model fit statistics
        print_or_write("Model Fit Statistics:", f)
        print_or_write(f"  Observations: {int(model.nobs)}", f)
        print_or_write(f"  Deviance: {model.deviance:.4f}", f)
        print_or_write(f"  Pearson chi2: {model.pearson_chi2:.4f}", f)
        print_or_write(f"  Log-Likelihood: {model.llf:.2f}", f)
        print_or_write(f"  AIC: {model.aic:.2f}", f)
        print_or_write(f"  BIC: {model.bic:.2f}", f)
        # Compute McFadden's pseudo R-squared manually for GLM
        if hasattr(model, 'llnull') and model.llnull != 0:
            pseudo_r2 = 1 - model.llf / model.llnull
            print_or_write(f"  Pseudo R-squared (McFadden): {pseudo_r2:.4f}", f)
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
        
        # Plot p_left vs predictor (no jitter needed - it's already continuous)
        ax.scatter(df[pred], df['p_left'], alpha=0.5, s=20)
        
        # Add GLM fit curve
        x_range = np.linspace(df[pred].min(), df[pred].max(), 100)
        
        # Fit GLM with logit link for this predictor
        # Clamp p_left to avoid 0/1
        y = np.clip(df['p_left'].values, 0.001, 0.999)
        X = sm.add_constant(df[pred].values)
        try:
            model = GLM(y, X, family=Binomial(link=Logit())).fit(disp=0)
            # Predict on the x_range
            X_pred = sm.add_constant(x_range)
            y_pred = model.predict(X_pred)
            ax.plot(x_range, y_pred, 'r-', linewidth=2, label='GLM fit')
        except Exception:
            pass  # Skip fit line if model fails
        
        ax.set_xlabel(pred)
        ax.set_ylabel('P(left)')
        ax.set_ylim(-0.05, 1.05)
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
    p_left_std = df['p_left'].std()
    if p_left_std < 0.001:
        print("ERROR: No variance in outcome variable (all p_left values are nearly identical).")
        print(f"  p_left std: {p_left_std}")
        print("\nLinear regression requires variance in the outcome.")
        print("Please run with more samples or different parameter ranges.")
        print()
        print("Parameter summary statistics:")
        print(df[['max_steps', 'beta_h', 'gamma_h', 'gamma_r', 'zeta', 'eta', 'xi', 'p_left']].describe())
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
