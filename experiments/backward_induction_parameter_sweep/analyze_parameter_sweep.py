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
    python experiments/backward_induction_parameter_sweep/analyze_parameter_sweep.py outputs/parameter_sweep/results.csv
    python experiments/backward_induction_parameter_sweep/analyze_parameter_sweep.py outputs/parameter_sweep/results.csv --interactions
    python experiments/backward_induction_parameter_sweep/analyze_parameter_sweep.py outputs/parameter_sweep/results.csv --output analysis_results.txt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Any

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
except ImportError as exc:
    HAVE_STATSMODELS = False
    raise ImportError(
        "statsmodels is required to run analyze_parameter_sweep.py but is not installed. "
        "Please install dependencies first, for example with:\n"
        "    pip install -r requirements.txt\n"
        "or:\n"
        "    pip install statsmodels"
    ) from exc

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
    
    # Print human positions if available
    if 'h1_pos' in df_valid.columns and 'h2_pos' in df_valid.columns:
        h1_positions = df_valid['h1_pos'].dropna().unique()
        h2_positions = df_valid['h2_pos'].dropna().unique()
        print()
        print("Human positions in initial state:")
        print(f"  h1 (human_agent_indices[0]): {', '.join(str(p) for p in h1_positions)}")
        print(f"  h2 (human_agent_indices[1]): {', '.join(str(p) for p in h2_positions)}")
    
    print()
    
    return df_valid


def compute_derived_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived outcome variables from raw metrics.
    
    Adds columns for:
    - pi_r_ratio: pi_r(turn_left) / (pi_r(turn_left) + pi_r(turn_right)) [proportion in 0,1]
    - Q_r_ratio: |Q_r(turn_left)| / (|Q_r(turn_left)| + |Q_r(turn_right)|) [proportion in 0,1]
    - X_h_ratio: X_h1 / (X_h1 + X_h2) [proportion in 0,1]
    - -log2(-V_r): -log2(-V_r) (V_r is negative, so -V_r > 0)
    - log2_X_h1: log2(X_h1)
    - log2_X_h2: log2(X_h2)
    - log2_X_h_diff: log2(X_h1) - log2(X_h2) = log2(X_h1 / X_h2)
    
    Args:
        df: DataFrame with raw results
        
    Returns:
        DataFrame with additional derived columns
    """
    df = df.copy()
    
    # pi_r(turn_left) / (pi_r(turn_left) + pi_r(turn_right)) - raw proportion
    if 'pi_r_turn_left' in df.columns and 'pi_r_turn_right' in df.columns:
        pi_sum = df['pi_r_turn_left'] + df['pi_r_turn_right']
        # Avoid division by zero, clamp to (0,1) for GLM
        pi_ratio = np.where(pi_sum > 0, df['pi_r_turn_left'] / pi_sum, 0.5)
        df['pi_r_ratio'] = np.clip(pi_ratio, 0.001, 0.999)
    
    # For Q_r, both values are negative, so we compute ratio of magnitudes:
    # |Q_r(turn_left)| / (|Q_r(turn_left)| + |Q_r(turn_right)|)
    if 'Q_r_turn_left' in df.columns and 'Q_r_turn_right' in df.columns:
        # Use absolute values since Q_r is always negative
        Q_left = np.abs(df['Q_r_turn_left'])
        Q_right = np.abs(df['Q_r_turn_right'])
        Q_sum = Q_left + Q_right
        Q_ratio = np.where(Q_sum > 0, Q_left / Q_sum, 0.5)
        df['Q_r_ratio'] = np.clip(Q_ratio, 0.001, 0.999)
    
    # X_h1 / (X_h1 + X_h2)
    if 'X_h1' in df.columns and 'X_h2' in df.columns:
        X_sum = df['X_h1'] + df['X_h2']
        X_ratio = np.where(X_sum > 0, df['X_h1'] / X_sum, 0.5)
        df['X_h_ratio'] = np.clip(X_ratio, 0.001, 0.999)
    
    # -log2(-V_r) - V_r is negative, so -V_r > 0
    if 'V_r' in df.columns:
        # Avoid log of zero
        V_r_neg = -df['V_r']
        df['-log2(-V_r)'] = np.where(V_r_neg > 0, -np.log2(V_r_neg), np.nan)
    
    # log2(X_h1) and log2(X_h2)
    if 'X_h1' in df.columns:
        df['log2_X_h1'] = np.where(df['X_h1'] > 0, np.log2(df['X_h1']), np.nan)
    if 'X_h2' in df.columns:
        df['log2_X_h2'] = np.where(df['X_h2'] > 0, np.log2(df['X_h2']), np.nan)
    
    # log2(X_h1) - log2(X_h2) = log2(X_h1 / X_h2)
    if 'X_h1' in df.columns and 'X_h2' in df.columns:
        # Only compute where both are positive
        valid = (df['X_h1'] > 0) & (df['X_h2'] > 0)
        df['log2_X_h_diff'] = np.where(valid, np.log2(df['X_h1']) - np.log2(df['X_h2']), np.nan)
    
    return df


def compute_univariate_ols(df: pd.DataFrame, outcome: str, predictors: List[str]) -> pd.DataFrame:
    """
    Compute univariate OLS regression for each predictor on a continuous outcome.
    
    Args:
        df: DataFrame with results
        outcome: Name of the outcome variable column
        predictors: List of predictor variable names
        
    Returns:
        DataFrame with coefficients, p-values, and confidence intervals
    """
    import statsmodels.api as sm
    
    results = []
    
    # Filter out rows with missing outcome
    df_valid = df[df[outcome].notna()].copy()
    if len(df_valid) == 0:
        print(f"  No valid data for outcome '{outcome}'")
        return pd.DataFrame()
    
    y = df_valid[outcome]
    
    for pred in predictors:
        # Skip predictors with zero variance
        if df_valid[pred].std() == 0:
            print(f"  Skipping '{pred}': zero variance")
            results.append({
                'predictor': pred,
                'coefficient': np.nan,
                'std_err': np.nan,
                't_value': np.nan,
                'p_value': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'significant': False
            })
            continue
        
        try:
            X = sm.add_constant(df_valid[pred])
            model = sm.OLS(y, X).fit()
            
            coef = model.params[pred]
            pval = model.pvalues[pred]
            ci_low, ci_high = model.conf_int().loc[pred]
            
            results.append({
                'predictor': pred,
                'coefficient': coef,
                'std_err': model.bse[pred],
                't_value': model.tvalues[pred],
                'p_value': pval,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'significant': pval < 0.05
            })
        except Exception as e:
            results.append({
                'predictor': pred,
                'coefficient': np.nan,
                'std_err': np.nan,
                't_value': np.nan,
                'p_value': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'significant': False
            })
    
    return pd.DataFrame(results)


def fit_full_ols_model(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Tuple[Any, pd.DataFrame]:
    """
    Fit full OLS model with all predictors on a continuous outcome.
    
    Args:
        df: DataFrame with results
        outcome: Name of the outcome variable column  
        predictors: List of predictor variable names
        
    Returns:
        Tuple of (fitted model, summary DataFrame)
    """
    import statsmodels.api as sm
    
    # Filter out rows with missing outcome
    df_valid = df[df[outcome].notna()].copy()
    if len(df_valid) == 0:
        print(f"  No valid data for outcome '{outcome}'")
        return None, None
    
    y = df_valid[outcome]
    
    # Filter out zero-variance predictors
    valid_predictors = [p for p in predictors if df_valid[p].std() > 0]
    
    X = sm.add_constant(df_valid[valid_predictors])
    
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"ERROR: Could not fit OLS model - {e}")
        return None, None
    
    summary = pd.DataFrame({
        'coefficient': model.params,
        'std_err': model.bse,
        't_value': model.tvalues,
        'p_value': model.pvalues,
        'significant': model.pvalues < 0.05
    })
    
    return model, summary


def print_ols_summary(outcome_name: str, model, summary: pd.DataFrame, output_file=None):
    """
    Print a formatted summary of an OLS model.
    """
    def print_or_write(text, file=None):
        print(text)
        if file:
            file.write(text + '\n')
    
    f = open(output_file, 'a') if output_file else None
    
    try:
        print_or_write(f"\n{'='*80}", f)
        print_or_write(f"OLS REGRESSION: {outcome_name}", f)
        print_or_write("="*80, f)
        
        print_or_write(f"  Observations: {int(model.nobs)}", f)
        print_or_write(f"  R-squared: {model.rsquared:.4f}", f)
        print_or_write(f"  Adj. R-squared: {model.rsquared_adj:.4f}", f)
        print_or_write(f"  F-statistic: {model.fvalue:.4f} (p={model.f_pvalue:.4f})", f)
        print_or_write("", f)
        
        header = f"{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8} {'Sig':>5}"
        print_or_write(header, f)
        print_or_write("-" * len(header), f)
        
        for var, row in summary.iterrows():
            sig_marker = "***" if row['p_value'] < 0.001 else \
                        "**" if row['p_value'] < 0.01 else \
                        "*" if row['p_value'] < 0.05 else ""
            line = f"{var:<20} {row['coefficient']:>10.4f} {row['std_err']:>10.4f} " \
                  f"{row['t_value']:>8.3f} {row['p_value']:>8.4f} {sig_marker:>5}"
            print_or_write(line, f)
        
        print_or_write("", f)
        print_or_write("Significance: *** p<0.001, ** p<0.01, * p<0.05", f)
        
    finally:
        if f:
            f.close()


def compute_univariate_glm_proportion(df: pd.DataFrame, outcome: str, predictors: List[str]) -> pd.DataFrame:
    """
    Compute univariate GLM (binomial family, logit link) for each predictor on a proportion outcome.
    
    Args:
        df: DataFrame with results
        outcome: Name of the outcome variable column (proportion in [0,1])
        predictors: List of predictor variable names
        
    Returns:
        DataFrame with coefficients, p-values, and odds ratios
    """
    results = []
    
    # Filter out rows with missing outcome
    df_valid = df[df[outcome].notna()].copy()
    if len(df_valid) == 0:
        print(f"  No valid data for outcome '{outcome}'")
        return pd.DataFrame()
    
    y = df_valid[outcome]
    
    for pred in predictors:
        # Skip predictors with zero variance
        if df_valid[pred].std() == 0:
            print(f"  Skipping '{pred}': zero variance")
            results.append({
                'predictor': pred,
                'coefficient': np.nan,
                'odds_ratio': np.nan,
                'std_err': np.nan,
                'z_value': np.nan,
                'p_value': np.nan,
                'significant': False
            })
            continue
        
        try:
            X = sm.add_constant(df_valid[pred])
            model = GLM(y, X, family=Binomial(link=Logit())).fit()
            
            coef = model.params[pred]
            pval = model.pvalues[pred]
            odds_ratio = np.exp(coef)
            
            results.append({
                'predictor': pred,
                'coefficient': coef,
                'odds_ratio': odds_ratio,
                'std_err': model.bse[pred],
                'z_value': model.tvalues[pred],
                'p_value': pval,
                'significant': pval < 0.05
            })
        except Exception as e:
            results.append({
                'predictor': pred,
                'coefficient': np.nan,
                'odds_ratio': np.nan,
                'std_err': np.nan,
                'z_value': np.nan,
                'p_value': np.nan,
                'significant': False
            })
    
    return pd.DataFrame(results)


def fit_full_glm_proportion(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Tuple[Any, pd.DataFrame]:
    """
    Fit full GLM (binomial family, logit link) with all predictors on a proportion outcome.
    
    Args:
        df: DataFrame with results
        outcome: Name of the outcome variable column (proportion in [0,1])
        predictors: List of predictor variable names
        
    Returns:
        Tuple of (fitted model, summary DataFrame)
    """
    # Filter out rows with missing outcome
    df_valid = df[df[outcome].notna()].copy()
    if len(df_valid) == 0:
        print(f"  No valid data for outcome '{outcome}'")
        return None, None
    
    y = df_valid[outcome]
    
    # Filter out zero-variance predictors
    valid_predictors = [p for p in predictors if df_valid[p].std() > 0]
    
    X = sm.add_constant(df_valid[valid_predictors])
    
    try:
        model = GLM(y, X, family=Binomial(link=Logit())).fit()
    except Exception as e:
        print(f"ERROR: Could not fit GLM model - {e}")
        return None, None
    
    summary = pd.DataFrame({
        'coefficient': model.params,
        'std_err': model.bse,
        'z_value': model.tvalues,
        'p_value': model.pvalues,
        'odds_ratio': np.exp(model.params),
        'significant': model.pvalues < 0.05
    })
    
    return model, summary


def print_glm_summary(outcome_name: str, model, summary: pd.DataFrame, output_file=None):
    """
    Print a formatted summary of a GLM model for proportions.
    """
    def print_or_write(text, file=None):
        print(text)
        if file:
            file.write(text + '\n')
    
    f = open(output_file, 'a') if output_file else None
    
    try:
        print_or_write(f"\n{'='*80}", f)
        print_or_write(f"GLM (Binomial/Logit): {outcome_name}", f)
        print_or_write("="*80, f)
        
        print_or_write(f"  Observations: {int(model.nobs)}", f)
        print_or_write(f"  Deviance: {model.deviance:.4f}", f)
        print_or_write(f"  Pearson chi2: {model.pearson_chi2:.4f}", f)
        print_or_write(f"  Log-Likelihood: {model.llf:.2f}", f)
        if hasattr(model, 'llnull') and model.llnull != 0:
            pseudo_r2 = 1 - model.llf / model.llnull
            print_or_write(f"  Pseudo R-squared: {pseudo_r2:.4f}", f)
        print_or_write("", f)
        
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
        print_or_write("Interpretation: Odds ratio > 1 means increasing predictor increases the proportion", f)
        
    finally:
        if f:
            f.close()


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
        # Skip predictors with zero variance (constant values)
        if df[pred].std() == 0:
            print(f"  Skipping '{pred}': zero variance (all values = {df[pred].iloc[0]})")
            results.append({
                'predictor': pred,
                'coefficient': np.nan,
                'odds_ratio': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'p_value': np.nan,
                'significant': False
            })
            continue
            
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
    
    # Filter out zero-variance predictors
    valid_predictors = [p for p in predictors if df[p].std() > 0]
    skipped = set(predictors) - set(valid_predictors)
    if skipped:
        print(f"Skipping zero-variance predictors: {skipped}")
    
    # Build design matrix
    X = df[valid_predictors].copy()
    
    if include_interactions:
        # Add interaction terms (only if both predictors have variance)
        if 'beta_h' in valid_predictors and 'gamma_h' in valid_predictors:
            X['beta_h_x_gamma_h'] = df['beta_h'] * df['gamma_h']
        if 'gamma_r' in valid_predictors and 'gamma_h' in valid_predictors:
            X['gamma_r_x_gamma_h'] = df['gamma_r'] * df['gamma_h']
        if 'zeta' in valid_predictors and 'xi' in valid_predictors:
            X['zeta_x_xi'] = df['zeta'] * df['xi']
        if 'max_steps' in valid_predictors and 'beta_h' in valid_predictors:
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


def create_scatterplot_matrices(df: pd.DataFrame,
                                predictors: List[str],
                                outcomes: List[Tuple[str, str, str]],
                                output_dir: str):
    """
    Create three bivariate scatterplot matrices:
    1. Regressors vs Regressants
    2. Regressors vs Regressors
    3. Regressants vs Regressants
    
    Args:
        df: DataFrame with results
        predictors: List of predictor (regressor) variable names
        outcomes: List of (column_name, description, model_type) tuples for regressants
        output_dir: Directory to save plots
    """
    if not HAVE_MATPLOTLIB:
        print("Skipping scatterplot matrices (matplotlib not available)")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract outcome column names that exist in the dataframe
    outcome_cols = [col for col, _, _ in outcomes if col in df.columns and df[col].notna().any()]
    
    if len(outcome_cols) == 0:
        print("No valid outcome columns for scatterplot matrices")
        return
    
    # Filter predictors to those with variance
    valid_predictors = [p for p in predictors if p in df.columns and df[p].std() > 0]
    
    if len(valid_predictors) == 0:
        print("No valid predictors for scatterplot matrices")
        return
    
    print(f"\nCreating scatterplot matrices...")
    print(f"  Regressors: {valid_predictors}")
    print(f"  Regressants: {outcome_cols}")
    
    # 1. Regressors vs Regressants scatterplot matrix
    _create_cross_scatterplot_matrix(
        df, valid_predictors, outcome_cols,
        "Regressors vs Regressants",
        output_path / 'scatterplot_matrix_regressors_vs_regressants.png'
    )
    
    # 2. Regressors vs Regressors scatterplot matrix
    _create_symmetric_scatterplot_matrix(
        df, valid_predictors,
        "Regressors vs Regressors",
        output_path / 'scatterplot_matrix_regressors.png'
    )
    
    # 3. Regressants vs Regressants scatterplot matrix
    if len(outcome_cols) > 1:
        _create_symmetric_scatterplot_matrix(
            df, outcome_cols,
            "Regressants vs Regressants",
            output_path / 'scatterplot_matrix_regressants.png'
        )
    else:
        print(f"  Skipping regressants vs regressants (only {len(outcome_cols)} outcome column)")


def _create_cross_scatterplot_matrix(df: pd.DataFrame,
                                      x_vars: List[str],
                                      y_vars: List[str],
                                      title: str,
                                      output_path: Path):
    """
    Create a scatterplot matrix with x_vars on columns and y_vars on rows.
    
    Args:
        df: DataFrame with data
        x_vars: Variables for x-axis (columns)
        y_vars: Variables for y-axis (rows)
        title: Plot title
        output_path: Path to save the figure
    """
    n_cols = len(x_vars)
    n_rows = len(y_vars)
    
    # Calculate figure size based on number of variables
    fig_width = max(12, n_cols * 2)
    fig_height = max(10, n_rows * 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle single row/column case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, y_var in enumerate(y_vars):
        for j, x_var in enumerate(x_vars):
            ax = axes[i, j]
            
            # Get valid data (both x and y not NaN)
            mask = df[x_var].notna() & df[y_var].notna()
            x_data = df.loc[mask, x_var]
            y_data = df.loc[mask, y_var]
            
            if len(x_data) > 0:
                ax.scatter(x_data, y_data, alpha=0.1, s=15, c='steelblue')
                
                # Add LOWESS nonparametric regression line
                if len(x_data) > 10:
                    try:
                        from statsmodels.nonparametric.smoothers_lowess import lowess
                        # Sort by x for a clean line
                        lowess_result = lowess(y_data.values, x_data.values, frac=0.3, it=3)
                        ax.plot(lowess_result[:, 0], lowess_result[:, 1],
                               color='red', linewidth=1.5, alpha=0.8)
                    except Exception:
                        pass  # Skip if LOWESS fails
                
                # Add correlation coefficient
                if len(x_data) > 2:
                    corr, pval = stats.pearsonr(x_data, y_data)
                    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                    ax.text(0.05, 0.95, f'r={corr:.2f}{sig}',
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Only show x-axis label on bottom row
            if i == n_rows - 1:
                ax.set_xlabel(x_var, fontsize=8)
            else:
                ax.set_xticklabels([])
            
            # Only show y-axis label on first column
            if j == 0:
                ax.set_ylabel(y_var, fontsize=8)
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved {title} to {output_path}")


def _create_symmetric_scatterplot_matrix(df: pd.DataFrame,
                                          variables: List[str],
                                          title: str,
                                          output_path: Path):
    """
    Create a symmetric scatterplot matrix (variables vs themselves).
    Shows histograms on diagonal and scatterplots with correlations off-diagonal.
    
    Args:
        df: DataFrame with data
        variables: Variables to plot
        title: Plot title
        output_path: Path to save the figure
    """
    n_vars = len(variables)
    
    # Calculate figure size based on number of variables
    fig_size = max(10, n_vars * 1.8)
    
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(fig_size, fig_size))
    
    # Handle single variable case
    if n_vars == 1:
        axes = np.array([[axes]])
    
    for i, y_var in enumerate(variables):
        for j, x_var in enumerate(variables):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                valid_data = df[x_var].dropna()
                if len(valid_data) > 0:
                    ax.hist(valid_data, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
                ax.set_ylabel('')
            else:
                # Off-diagonal: scatterplot
                mask = df[x_var].notna() & df[y_var].notna()
                x_data = df.loc[mask, x_var]
                y_data = df.loc[mask, y_var]
                
                if len(x_data) > 0:
                    ax.scatter(x_data, y_data, alpha=0.1, s=15, c='steelblue')
                    
                    # Add correlation coefficient
                    if len(x_data) > 2:
                        corr, pval = stats.pearsonr(x_data, y_data)
                        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                        ax.text(0.05, 0.95, f'r={corr:.2f}{sig}',
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Only show x-axis label on bottom row
            if i == n_vars - 1:
                ax.set_xlabel(x_var, fontsize=8)
            else:
                ax.set_xticklabels([])
            
            # Only show y-axis label on first column
            if j == 0 and i != j:
                ax.set_ylabel(y_var, fontsize=8)
            elif j == 0 and i == j:
                ax.set_ylabel(y_var, fontsize=8)
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved {title} to {output_path}")


def run_analysis(df: pd.DataFrame, predictors: List[str], outcomes: List[Tuple[str, str, str]],
                  plots_dir: str, label: str = "FULL SAMPLE", output_file: str = None):
    """
    Run regressions and create scatterplot matrices for a given DataFrame.
    
    Args:
        df: DataFrame with results (already has derived columns)
        predictors: List of predictor variable names
        outcomes: List of (column, description, model_type) tuples
        plots_dir: Directory for output plots
        label: Label for this analysis subset
        output_file: Optional output file for text results
    """
    print()
    print("#" * 80)
    print(f"# {label} (N={len(df)})")
    print("#" * 80)
    
    if len(df) < 10:
        print(f"WARNING: Only {len(df)} samples in {label}. Skipping.")
        return
    
    for outcome_col, outcome_desc, model_type in outcomes:
        if outcome_col not in df.columns or df[outcome_col].isna().all():
            print(f"\n{'='*80}")
            print(f"SKIPPING: {outcome_desc}")
            print(f"  (Column '{outcome_col}' not available or all NaN)")
            print("="*80)
            continue
        
        # Check for variance
        outcome_std = df[outcome_col].std()
        if pd.isna(outcome_std) or outcome_std < 1e-10:
            print(f"\n{'='*80}")
            print(f"SKIPPING: {outcome_desc}")
            print(f"  (No variance in outcome: std={outcome_std})")
            print("="*80)
            continue
        
        print(f"\n{'='*80}")
        print(f"UNIVARIATE ANALYSIS [{label}]: {outcome_desc} [{model_type.upper()}]")
        print("="*80)
        print()
        
        # Summary statistics for this outcome
        print(f"  N valid: {df[outcome_col].notna().sum()}")
        print(f"  Mean: {df[outcome_col].mean():.6f}")
        print(f"  Std: {df[outcome_col].std():.6f}")
        print(f"  Range: [{df[outcome_col].min():.6f}, {df[outcome_col].max():.6f}]")
        print()
        
        if model_type == 'glm':
            # GLM for proportion outcomes
            univar_results = compute_univariate_glm_proportion(df, outcome_col, predictors)
            if len(univar_results) > 0:
                print(univar_results.to_string(index=False))
            print()
            
            # Full model
            model_fit, summary_fit = fit_full_glm_proportion(df, outcome_col, predictors)
            if model_fit is not None:
                print_glm_summary(outcome_desc + f' [{label}]', model_fit, summary_fit, output_file=output_file)
        else:
            # OLS for continuous outcomes
            univar_results = compute_univariate_ols(df, outcome_col, predictors)
            if len(univar_results) > 0:
                print(univar_results.to_string(index=False))
            print()
            
            # Full model
            model_fit, summary_fit = fit_full_ols_model(df, outcome_col, predictors)
            if model_fit is not None:
                print_ols_summary(outcome_desc + f' [{label}]', model_fit, summary_fit, output_file=output_file)
    
    # Create scatterplot matrices
    create_scatterplot_matrices(df, predictors, outcomes, plots_dir)


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
    parser.add_argument('--max_steps_threshold', type=int, default=10,
                       help='Threshold for splitting subsamples by max_steps (default: 10)')
    
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
    
    # Compute derived outcome variables
    df = compute_derived_outcomes(df)
    
    # Define predictors (exclude beta_r since it's fixed)
    predictors = ['max_steps', 'beta_h', 'gamma_h', 'gamma_r', 'zeta', 'eta', 'xi']
    
    # Define outcomes: (column, description, model_type)
    # model_type: 'glm' for proportions in [0,1], 'ols' for continuous
    outcomes = [
        ('pi_r_ratio', 'pi_r(turn_left) / (pi_r(turn_left) + pi_r(turn_right))', 'glm'),
        ('Q_r_ratio', '|Q_r(turn_left)| / (|Q_r(turn_left)| + |Q_r(turn_right)|)', 'glm'),
        ('-log2(-V_r)', '-log2(-V_r) at initial state', 'ols'),
        ('log2_X_h1', 'log2(X_h1) for h1 (first human)', 'ols'),
        ('log2_X_h2', 'log2(X_h2) for h2 (second human)', 'ols'),
        ('X_h_ratio', 'X_h1 / (X_h1 + X_h2)', 'glm'),
        ('log2_X_h_diff', 'log2(X_h1) - log2(X_h2) = log2(X_h1/X_h2)', 'ols'),
    ]
    
    # ========================================================================
    # Run analysis on FULL sample
    # ========================================================================
    run_analysis(df, predictors, outcomes, args.plots_dir, label="FULL SAMPLE", output_file=args.output)
    
    # ========================================================================
    # Run analysis on subsamples split by max_steps threshold
    # ========================================================================
    threshold = args.max_steps_threshold
    
    df_low = df[df['max_steps'] < threshold].copy()
    df_high = df[df['max_steps'] > threshold].copy()
    
    print(f"\n\nSplitting by max_steps threshold = {threshold}:")
    print(f"  max_steps < {threshold}: {len(df_low)} samples")
    print(f"  max_steps > {threshold}: {len(df_high)} samples")
    
    plots_dir_low = str(Path(args.plots_dir) / f'max_steps_lt_{threshold}')
    plots_dir_high = str(Path(args.plots_dir) / f'max_steps_gt_{threshold}')
    
    run_analysis(df_low, predictors, outcomes, plots_dir_low,
                 label=f"max_steps < {threshold}", output_file=args.output)
    
    run_analysis(df_high, predictors, outcomes, plots_dir_high,
                 label=f"max_steps > {threshold}", output_file=args.output)
    
    if args.output:
        print(f"\nDetailed results saved to: {args.output}")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
