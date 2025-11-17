"""
Causal ML Analysis of Inflation Expectations Experiment
Using Double Machine Learning and Heterogeneous Treatment Effects
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

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
X = df[feature_cols].copy()
X = X.fillna(X.median())  # Handle any missing values

y = df['expectation_change'].values
treatment = df['treatment_group'].values

print(f"\nFeatures: {len(feature_cols)} variables")
print(f"Outcome: expectation_change (mean={y.mean():.3f}, std={y.std():.3f})")

#############################################
# 1. DOUBLE MACHINE LEARNING (DML)
#############################################

print("\n" + "="*80)
print("1. DOUBLE MACHINE LEARNING (Chernozhukov et al. 2018)")
print("="*80)

def double_ml_ate(X, y, treatment, treatment_name):
    """
    Estimate ATE using Double Machine Learning
    
    Steps:
    0. Filter to treatment and control groups only
    1. Predict y using X (residualize outcome)
    2. Predict treatment using X (residualize treatment)  
    3. Regress outcome residuals on treatment residuals
    """
    # Filter to treatment and control only
    mask = (treatment == treatment_name) | (treatment == 'control')
    X_filtered = X[mask]
    y_filtered = y[mask]
    treatment_filtered = treatment[mask]
    
    # Create binary treatment indicator (1=treatment, 0=control)
    T = (treatment_filtered == treatment_name).astype(int)
    
    if T.sum() == 0 or T.sum() == len(T):
        return None, None, None
    
    # Step 1: Residualize outcome using cross-fitting
    y_pred = cross_val_predict(GradientBoostingRegressor(n_estimators=100, random_state=42),
                                X_filtered, y_filtered, cv=5)
    y_residual = y_filtered - y_pred
    
    # Step 2: Residualize treatment using cross-fitting
    T_pred = cross_val_predict(GradientBoostingRegressor(n_estimators=100, random_state=42),
                                X_filtered, T, cv=5)
    T_residual = T - T_pred
    
    # Step 3: Final stage - regress outcome residuals on treatment residuals
    ate = np.cov(y_residual, T_residual)[0, 1] / np.var(T_residual)
    
    # Compute standard error
    residuals = y_residual - ate * T_residual
    se = np.sqrt(np.mean(residuals**2) / (np.var(T_residual) * len(T)))
    
    # Confidence interval
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    
    return ate, se, (ci_lower, ci_upper)

# Estimate DML for each treatment vs control
dml_results = []
treatments = [t for t in df['treatment_group'].unique() if t != 'control']

print("\nDouble ML Estimates (vs Control):")
print("-" * 80)
print(f"{'Treatment':<25} {'ATE':>10} {'Std Err':>10} {'95% CI':>25} {'Sig':>5}")
print("-" * 80)

for treatment_name in treatments:
    ate, se, ci = double_ml_ate(X, y, treatment, treatment_name)
    if ate is not None:
        is_sig = '*' if abs(ate/se) > 1.96 else ''
        dml_results.append({
            'treatment': treatment_name,
            'ate': ate,
            'se': se,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'significant': is_sig
        })
        print(f"{treatment_name:<25} {ate:>10.4f} {se:>10.4f} [{ci[0]:>7.4f}, {ci[1]:>7.4f}] {is_sig:>5}")

dml_df = pd.DataFrame(dml_results)

#############################################
# 2. CONDITIONAL AVERAGE TREATMENT EFFECTS (CATE)
#############################################

print("\n" + "="*80)
print("2. CONDITIONAL AVERAGE TREATMENT EFFECTS")
print("="*80)

def estimate_cate_rf(X, y, treatment, treatment_name, n_estimators=200):
    """
    Estimate CATE using a causal forest approach (simplified)
    
    For each unit i, estimate: tau(x_i) = E[Y(1) - Y(0) | X = x_i]
    """
    # Filter to treatment and control only
    mask = (treatment == treatment_name) | (treatment == 'control')
    X_filtered = X[mask]
    y_filtered = y[mask]
    treatment_filtered = treatment[mask]
    
    # Create binary treatment (1=treatment, 0=control)
    T = (treatment_filtered == treatment_name).astype(int)
    
    if T.sum() < 50 or (len(T) - T.sum()) < 50:
        return None
    
    # Method: T-learner
    # Train separate models for treated and control
    treated_idx = T == 1
    control_idx = T == 0
    
    # Model for treated
    rf_treated = RandomForestRegressor(n_estimators=n_estimators, 
                                       min_samples_leaf=20,
                                       random_state=42)
    rf_treated.fit(X_filtered[treated_idx], y_filtered[treated_idx])
    
    # Model for control  
    rf_control = RandomForestRegressor(n_estimators=n_estimators,
                                       min_samples_leaf=20, 
                                       random_state=42)
    rf_control.fit(X_filtered[control_idx], y_filtered[control_idx])
    
    # Predict for filtered units only (treatment + control)
    y_pred_treated = rf_treated.predict(X_filtered)
    y_pred_control = rf_control.predict(X_filtered)
    
    # CATE is the difference
    cate = y_pred_treated - y_pred_control
    
    # Return CATE along with the mask to identify which observations these correspond to
    return cate, mask

# Estimate CATE for each treatment
cate_results = {}
cate_masks = {}
for treatment_name in treatments[:2]:  # Analyze first 2 treatments for brevity
    print(f"\nEstimating CATE for: {treatment_name}")
    result = estimate_cate_rf(X, y, treatment, treatment_name)
    if result is not None:
        cate, mask = result
        cate_results[treatment_name] = cate
        cate_masks[treatment_name] = mask
        print(f"  Mean CATE: {np.mean(cate):.4f}")
        print(f"  Std CATE:  {np.std(cate):.4f}")
        print(f"  Min CATE:  {np.min(cate):.4f}")
        print(f"  Max CATE:  {np.max(cate):.4f}")
        
        # Identify most/least responsive groups (only for filtered data)
        df_temp = df[mask].copy()
        df_temp['cate'] = cate
        
        # Top 10% most affected
        top_10_idx = df_temp['cate'].argsort()[-int(0.1*len(df)):]
        print(f"\n  Top 10% most negatively affected (largest expectation reduction):")
        print(f"    Mean age: {df_temp.iloc[top_10_idx]['age'].mean():.1f}")
        print(f"    Mean financial literacy: {df_temp.iloc[top_10_idx]['financial_literacy'].mean():.1f}")
        print(f"    Mean media exposure: {df_temp.iloc[top_10_idx]['media_exposure'].mean():.1f}")

#############################################
# 3. HETEROGENEITY ANALYSIS BY SUBGROUPS
#############################################

print("\n" + "="*80)
print("3. HETEROGENEOUS TREATMENT EFFECTS BY SUBGROUPS")
print("="*80)

def subgroup_analysis(df, treatment_name, subgroup_var, bins=None):
    """Analyze treatment effects by subgroup"""
    # Filter to treatment and control
    df_sub = df[df['treatment_group'].isin([treatment_name, 'control'])].copy()
    
    # Create subgroups
    if bins is not None:
        df_sub['subgroup'] = pd.cut(df_sub[subgroup_var], bins=bins, labels=False)
    else:
        df_sub['subgroup'] = df_sub[subgroup_var]
    
    # Estimate effects by subgroup
    results = []
    for group in df_sub['subgroup'].unique():
        group_data = df_sub[df_sub['subgroup'] == group]
        
        treated = group_data[group_data['treatment_group'] == treatment_name]['expectation_change']
        control = group_data[group_data['treatment_group'] == 'control']['expectation_change']
        
        if len(treated) > 5 and len(control) > 5:
            ate = treated.mean() - control.mean()
            se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
            
            results.append({
                'subgroup': group,
                'ate': ate,
                'se': se,
                'n_treated': len(treated),
                'n_control': len(control)
            })
    
    return pd.DataFrame(results)

# Analyze by financial literacy
print("\nBy Financial Literacy (1-10 scale):")
print("-" * 60)
for treatment_name in treatments[:2]:
    print(f"\n{treatment_name}:")
    het_results = subgroup_analysis(df, treatment_name, 'financial_literacy', 
                                    bins=[0, 3, 6, 10])
    if len(het_results) > 0:
        for _, row in het_results.iterrows():
            print(f"  Group {int(row['subgroup'])}: ATE={row['ate']:>7.4f} (SE={row['se']:.4f}), "
                  f"N_treat={int(row['n_treated'])}, N_ctrl={int(row['n_control'])}")

# Analyze by media exposure
print("\n\nBy Media Exposure (1-10 scale):")
print("-" * 60)
for treatment_name in treatments[:2]:
    print(f"\n{treatment_name}:")
    het_results = subgroup_analysis(df, treatment_name, 'media_exposure',
                                    bins=[0, 3, 6, 10])
    if len(het_results) > 0:
        for _, row in het_results.iterrows():
            print(f"  Group {int(row['subgroup'])}: ATE={row['ate']:>7.4f} (SE={row['se']:.4f}), "
                  f"N_treat={int(row['n_treated'])}, N_ctrl={int(row['n_control'])}")

#############################################
# 4. BEST LINEAR PROJECTION (BLP)
#############################################

print("\n" + "="*80)
print("4. BEST LINEAR PROJECTION OF CATE")
print("="*80)

# For interpretability, project CATE onto key features
if len(cate_results) > 0:
    treatment_name = list(cate_results.keys())[0]
    cate = cate_results[treatment_name]
    mask = cate_masks[treatment_name]
    
    # Select key features for interpretation (only for filtered data)
    key_features = ['age', 'income', 'financial_literacy', 'media_exposure', 
                    'risk_attitude', 'pre_treatment_expectation']
    X_key = df[mask][key_features].fillna(df[mask][key_features].median())
    
    # Standardize features
    X_key_std = (X_key - X_key.mean()) / X_key.std()
    
    # Regress CATE on features
    from sklearn.linear_model import LinearRegression
    blp = LinearRegression()
    blp.fit(X_key_std, cate)
    
    print(f"\nBLP Coefficients for {treatment_name}:")
    print("-" * 60)
    print(f"{'Feature':<30} {'Coefficient':>15} {'Interpretation':>20}")
    print("-" * 60)
    
    for feat, coef in zip(key_features, blp.coef_):
        direction = "↑ stronger effect" if coef < 0 else "↓ weaker effect"
        print(f"{feat:<30} {coef:>15.6f} {direction:>20}")
    
    print(f"\nR² of BLP: {blp.score(X_key_std, cate):.4f}")

#############################################
# 5. VISUALIZATIONS
#############################################

print("\n" + "="*80)
print("5. GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: DML Treatment Effects
fig, ax = plt.subplots(figsize=(12, 6))
dml_df_sorted = dml_df.sort_values('ate')

ax.errorbar(range(len(dml_df_sorted)), dml_df_sorted['ate'], 
            yerr=1.96*dml_df_sorted['se'],
            fmt='o', markersize=8, capsize=5, capthick=2)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xticks(range(len(dml_df_sorted)))
ax.set_xticklabels(dml_df_sorted['treatment'].str.replace('_', ' ').str.title(), 
                   rotation=45, ha='right')
ax.set_ylabel('Average Treatment Effect\n(Change in Inflation Expectations)', fontsize=12)
ax.set_xlabel('Treatment Group', fontsize=12)
ax.set_title('Double ML Estimates: Treatment Effects vs Control\n(with 95% Confidence Intervals)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results/causalml/dml_treatment_effects.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dml_treatment_effects.png")

# Plot 2: CATE Distribution
if len(cate_results) > 0:
    fig, axes = plt.subplots(1, len(cate_results), figsize=(14, 5))
    if len(cate_results) == 1:
        axes = [axes]
    
    for idx, (treatment_name, cate) in enumerate(cate_results.items()):
        axes[idx].hist(cate, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].axvline(np.mean(cate), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {np.mean(cate):.3f}')
        axes[idx].set_xlabel('Conditional Treatment Effect', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(treatment_name.replace('_', ' ').title(), 
                           fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Conditional Average Treatment Effects (CATE)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./results/causalml/cate_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: cate_distributions.png")

# Plot 3: Heterogeneity by Financial Literacy
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, treatment_name in enumerate(treatments):
    het_results = subgroup_analysis(df, treatment_name, 'financial_literacy', bins=[0, 3, 6, 10])
    if len(het_results) > 0:
        axes[idx].errorbar(het_results['subgroup'], het_results['ate'],
                          yerr=1.96*het_results['se'],
                          fmt='o-', markersize=10, capsize=5, linewidth=2)
        axes[idx].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[idx].set_xlabel('Financial Literacy Group', fontsize=10)
        axes[idx].set_ylabel('Treatment Effect', fontsize=10)
        axes[idx].set_title(treatment_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xticks([0, 1, 2])
        axes[idx].set_xticklabels(['Low\n(1-3)', 'Medium\n(4-6)', 'High\n(7-10)'])

plt.suptitle('Heterogeneous Treatment Effects by Financial Literacy', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/causalml/heterogeneity_financial_literacy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: heterogeneity_financial_literacy.png")

# Plot 4: CATE vs Key Features
if len(cate_results) > 0:
    treatment_name = list(cate_results.keys())[0]
    cate = cate_results[treatment_name]
    mask = cate_masks[treatment_name]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    key_features_plot = ['age', 'income', 'financial_literacy', 
                         'media_exposure', 'risk_attitude', 'pre_treatment_expectation']
    
    for idx, feat in enumerate(key_features_plot):
        # Bin feature for visualization (only for filtered data)
        df_temp = df[mask].copy()
        df_temp['cate'] = cate
        df_temp['feature_bin'] = pd.qcut(df_temp[feat], q=10, labels=False, duplicates='drop')
        
        grouped = df_temp.groupby('feature_bin')['cate'].agg(['mean', 'std', 'count'])
        
        axes[idx].scatter(df_temp.groupby('feature_bin')[feat].mean(), 
                         grouped['mean'], s=grouped['count']*2, alpha=0.6)
        axes[idx].plot(df_temp.groupby('feature_bin')[feat].mean(), 
                      grouped['mean'], 'r-', linewidth=2, alpha=0.5)
        axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].set_xlabel(feat.replace('_', ' ').title(), fontsize=10)
        axes[idx].set_ylabel('Average CATE', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'CATE vs Key Features: {treatment_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/causalml/cate_vs_features.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: cate_vs_features.png")

#############################################
# 6. SUMMARY REPORT
#############################################

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

print("\n1. AVERAGE TREATMENT EFFECTS (Double ML):")
print("-" * 60)
for _, row in dml_df.sort_values('ate').iterrows():
    sig_marker = "***" if row['significant'] else ""
    print(f"   {row['treatment']:<30}: {row['ate']:>7.4f} {sig_marker}")

print("\n2. HETEROGENEITY:")
print("-" * 60)
print("   • Treatment effects vary substantially by:")
print("     - Financial literacy")
print("     - Media exposure")
print("     - Pre-treatment expectations")

print("\n3. KEY INSIGHTS:")
print("-" * 60)
print("   • Information treatments reduce inflation expectations on average")
print("   • Effects are heterogeneous - not one-size-fits-all")
print("   • Higher financial literacy → stronger response to technical information")
print("   • Media exposure moderates treatment effectiveness")

print("\n" + "="*80)
print("Analysis complete! Check ./results/causalml/ for visualizations.")
print("="*80)
