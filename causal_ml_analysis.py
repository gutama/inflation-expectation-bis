"""
Causal ML Analysis of Inflation Expectations Experiment
Using Double Machine Learning (DoubleML Package) and Heterogeneous Treatment Effects
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.base import clone
import doubleml as dml
import warnings
import os

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create results directory if not exists
os.makedirs('./results/causalml', exist_ok=True)

#############################################
# LOAD AND PREPARE DATA
#############################################

print("="*80)
print("CAUSAL ML ANALYSIS: Inflation Expectations Experiment")
print("="*80)

# Load data
df = pd.read_csv('./results/gpt-4.1-mini-exclusive-treatment-run2/all_experiment_results_20250821_054934.csv')

print(f"\nDataset: {len(df)} observations across {df['quarter'].nunique()} quarters")
print(f"Treatment groups: {df['treatment_group'].nunique()}")

# Parse persona JSON to extract covariates
def parse_persona(persona_str):
    """Parse persona JSON string"""
    try:
        persona_dict = json.loads(persona_str.replace("'", '"'))
        return pd.Series(persona_dict)
    except:
        return pd.Series()

persona_df = df['persona'].apply(parse_persona)
df = pd.concat([df, persona_df], axis=1)

# Create clean feature set
feature_cols = ['age', 'income', 'financial_literacy', 'media_exposure', 
                'risk_attitude', 'expenditure', 'pre_treatment_expectation', 
                'pre_confidence']

# Create dummy variables for categorical features
categorical_cols = ['gender', 'education', 'region', 'province', 'urban_rural', 
                    'expenditure_bracket']

for col in categorical_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())

# Prepare final dataset
# Clean column names for DoubleML (no spaces or special chars preferred)
feature_cols = [c.replace(' ', '_').replace('-', '_') for c in feature_cols]
df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]

# Deduplicate columns (keep first instance)
df = df.loc[:, ~df.columns.duplicated()]

X_cols = sorted(list(set(feature_cols))) # Deduplicate feature list
# Ensure all X_cols exist in df
X_cols = [c for c in X_cols if c in df.columns]

y_col = 'expectation_change'
treatment_col = 'treatment_group'

# Handle missing values in features
df[X_cols] = df[X_cols].fillna(df[X_cols].median())

print(f"\nFeatures: {len(X_cols)} variables")
print(f"Outcome: {y_col} (mean={df[y_col].mean():.3f}, std={df[y_col].std():.3f})")

#############################################
# 1. DOUBLE MACHINE LEARNING (DoubleML Package)
#############################################

print("\n" + "="*80)
print("1. DOUBLE MACHINE LEARNING (using DoubleML package)")
print("="*80)

# Estimate DML for each treatment vs control
dml_results = []
treatments = [t for t in df['treatment_group'].unique() if t != 'control']

print("\nDouble ML Estimates (vs Control):")
print("-" * 80)
print(f"{'Treatment':<25} {'ATE':>10} {'Std Err':>10} {'95% CI':>25} {'Sig':>5}")
print("-" * 80)

for treatment_name in treatments:
    # 1. Prepare Data for Binary Comparison (Treatment vs Control)
    df_binary = df[df['treatment_group'].isin([treatment_name, 'control'])].copy()
    
    # Create binary treatment indicator (1=treatment, 0=control)
    df_binary['T'] = (df_binary['treatment_group'] == treatment_name).astype(int)
    
    # Initialize DoubleML Data Object
    dml_data = dml.DoubleMLData(df_binary,
                                y_col=y_col,
                                d_cols='T',
                                x_cols=X_cols)
    
    # 2. Define Learners
    # Learner for outcome regression E[Y|X]
    ml_g = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    # Learner for propensity score E[D|X]
    ml_m = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # 3. Initialize and Fit DoubleML IRM (Interactive Regression Model) model
    np.random.seed(42)
    dml_irm = dml.DoubleMLIRM(dml_data,
                              ml_g=ml_g,
                              ml_m=ml_m,
                              n_folds=5)
    
    dml_irm.fit()
    
    # Extract results
    ate = dml_irm.coef[0]
    se = dml_irm.se[0]
    ci = dml_irm.confint(level=0.95)
    ci_lower = ci.iloc[0, 0]
    ci_upper = ci.iloc[0, 1]
    pval = dml_irm.pval[0]
    
    is_sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    
    dml_results.append({
        'treatment': treatment_name,
        'ate': ate,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': pval < 0.05, 
        'pval': pval
    })
    
    print(f"{treatment_name:<25} {ate:>10.4f} {se:>10.4f} [{ci_lower:>7.4f}, {ci_upper:>7.4f}] {is_sig:>5}")



dml_df = pd.DataFrame(dml_results)

#############################################
# 1b. METHOD COMPARISON (Naive vs OLS vs DML)
#############################################

print("\n" + "="*80)
print("1b. METHOD COMPARISON: Naive vs OLS vs DML")
print("="*80)

import statsmodels.api as sm

comparison_results = []

print("\nComparison of Estimates and Standard Errors:")
print("-" * 100)
print(f"{'Treatment':<25} {'Method':<10} {'ATE':>10} {'SE':>10} {'t-stat':>8}")
print("-" * 100)

for treatment_name in treatments:
    # Prepare estimates
    estimates = {}
    
    # 1. Naive (Simple Difference in Means)
    df_binary = df[df['treatment_group'].isin([treatment_name, 'control'])].copy()
    treated = df_binary[df_binary['treatment_group'] == treatment_name][y_col]
    control = df_binary[df_binary['treatment_group'] == 'control'][y_col]
    
    naive_ate = treated.mean() - control.mean()
    naive_se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
    estimates['Naive'] = (naive_ate, naive_se)
    
    # 2. OLS (Linear Regression with Controls)
    # y ~ T + X
    # Ensure X is numeric
    X_ols = df_binary[X_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_ols = sm.add_constant(X_ols)
    T_ols = (df_binary['treatment_group'] == treatment_name).astype(int).values
    y_ols = df_binary[y_col].values
    
    # Concatenate T to X
    X_full = np.column_stack((T_ols, X_ols)).astype(float)
    
    model = sm.OLS(y_ols, X_full).fit()
    ols_ate = model.params[0] # T is the first column
    ols_se = model.bse[0]
    estimates['OLS'] = (ols_ate, ols_se)
    
    # 3. DML (Already calculated)
    dml_row = dml_df[dml_df['treatment'] == treatment_name].iloc[0]
    estimates['DML'] = (dml_row['ate'], dml_row['se'])
    
    # Print and Store
    for method in ['Naive', 'OLS', 'DML']:
        ate, se = estimates[method]
        t_stat = ate / se if se > 0 else 0
        print(f"{treatment_name:<25} {method:<10} {ate:>10.4f} {se:>10.4f} {t_stat:>8.2f}")
        
        comparison_results.append({
            'treatment': treatment_name,
            'method': method,
            'ate': ate,
            'se': se,
            't_stat': t_stat
        })
    print("-" * 100) # Separator between treatments

comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv('./results/causalml/method_comparison.csv', index=False)
print("\nComparison saved to ./results/causalml/method_comparison.csv")

#############################################
# 2. CONDITIONAL AVERAGE TREATMENT EFFECTS (CATE)
#############################################

print("\n" + "="*80)
print("2. CONDITIONAL AVERAGE TREATMENT EFFECTS")
print("="*80)

def estimate_cate_rf(df, X_cols, y_col, treatment_name, n_estimators=200):
    """
    Estimate CATE using a T-learner approach with Random Forests
    """
    # Filter to treatment and control
    mask = df['treatment_group'].isin([treatment_name, 'control'])
    df_filtered = df[mask].copy()
    
    X = df_filtered[X_cols].values
    y = df_filtered[y_col].values
    T = (df_filtered['treatment_group'] == treatment_name).astype(int).values
    
    if T.sum() < 50 or (len(T) - T.sum()) < 50:
        return None
    
    # Train separate models
    rf_treated = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=20, random_state=42)
    rf_control = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=20, random_state=42)
    
    rf_treated.fit(X[T==1], y[T==1])
    rf_control.fit(X[T==0], y[T==0])
    
    # Predict CATE
    cate = rf_treated.predict(X) - rf_control.predict(X)
    
    return cate, mask

# Estimate CATE for each treatment
cate_results = {}
cate_masks = {}

# Analyze first 2 treatments for detailed logical walkthrough
treatments_to_analyze = treatments[:2] if len(treatments) >= 2 else treatments

for treatment_name in treatments_to_analyze:
    print(f"\nEstimating CATE for: {treatment_name}")
    result = estimate_cate_rf(df, X_cols, y_col, treatment_name)
    if result is not None:
        cate, mask = result
        cate_results[treatment_name] = cate
        cate_masks[treatment_name] = mask # Series of booleans matching df index
        
        print(f"  Mean CATE: {np.mean(cate):.4f}")
        print(f"  Std CATE:  {np.std(cate):.4f}")
        
        # Identify most/least responsive groups
        df_temp = df[mask].copy()
        df_temp['cate'] = cate
        
        top_10_idx = df_temp['cate'].argsort()[:int(0.1*len(df_temp))] # Most negative (strongest effect)
        print(f"\n  Top 10% strongest responders (most negative change):")
        print(f"    Mean age: {df_temp.iloc[top_10_idx]['age'].mean():.1f}")
        print(f"    Mean financial literacy: {df_temp.iloc[top_10_idx]['financial_literacy'].mean():.1f}")

#############################################
# 3. HETEROGENEITY ANALYSIS BY SUBGROUPS
#############################################

print("\n" + "="*80)
print("3. HETEROGENEOUS TREATMENT EFFECTS BY SUBGROUPS")
print("="*80)

def subgroup_analysis(df, treatment_name, subgroup_var, bins=None):
    """Analyze treatment effects by subgroup"""
    df_sub = df[df['treatment_group'].isin([treatment_name, 'control'])].copy()
    
    if bins is not None:
        df_sub['subgroup'] = pd.cut(df_sub[subgroup_var], bins=bins, labels=False)
    else:
        df_sub['subgroup'] = df_sub[subgroup_var]
    
    results = []
    for group in sorted(df_sub['subgroup'].unique()):
        if pd.isna(group): continue
        
        group_data = df_sub[df_sub['subgroup'] == group]
        treated = group_data[group_data['treatment_group'] == treatment_name][y_col]
        control = group_data[group_data['treatment_group'] == 'control'][y_col]
        
        if len(treated) > 5 and len(control) > 5:
            ate_group = treated.mean() - control.mean()
            se_group = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
            
            results.append({
                'subgroup': group,
                'ate': ate_group,
                'se': se_group,
                'n_treated': len(treated),
                'n_control': len(control)
            })
            
    return pd.DataFrame(results)

# Analyze by financial literacy
print("\nBy Financial Literacy (Low 0-3, Med 4-6, High 7-10):")
print("-" * 60)
for treatment_name in treatments_to_analyze:
    print(f"\n{treatment_name}:")
    het_results = subgroup_analysis(df, treatment_name, 'financial_literacy', bins=[0, 3, 6, 10])
    if not het_results.empty:
        for _, row in het_results.iterrows():
            print(f"  Group {int(row['subgroup'])}: ATE={row['ate']:>7.4f} (SE={row['se']:.4f})")

#############################################
# 4. VISUALIZATIONS
#############################################

print("\n" + "="*80)
print("4. GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: DML Treatment Effects
fig, ax = plt.subplots(figsize=(12, 6))
dml_df_sorted = dml_df.sort_values('ate')

ax.errorbar(range(len(dml_df_sorted)), dml_df_sorted['ate'], 
            yerr=[dml_df_sorted['ate'] - dml_df_sorted['ci_lower'], 
                  dml_df_sorted['ci_upper'] - dml_df_sorted['ate']],
            fmt='o', markersize=8, capsize=5, capthick=2)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xticks(range(len(dml_df_sorted)))
ax.set_xticklabels(dml_df_sorted['treatment'].str.replace('_', ' ').str.title(), 
                   rotation=45, ha='right')
ax.set_ylabel('Average Treatment Effect\n(Change in Inflation Expectations)', fontsize=12)
ax.set_xlabel('Treatment Group', fontsize=12)
ax.set_title('Double ML Estimates (DoubleML Package)\nTreatment Effects vs Control (95% CI)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results/causalml/dml_doubleml_package_effects.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dml_doubleml_package_effects.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
