"""
Advanced Causal ML Analysis using EconML
Implements: DML, Causal Forest, DR-Learner, X-Learner
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV, LassoCV
import warnings
warnings.filterwarnings('ignore')

# EconML imports
from econml.dml import LinearDML, CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import XLearner, TLearner
from econml.inference import BootstrapInference

# Set style
sns.set_style("whitegrid")

#############################################
# LOAD AND PREPARE DATA
#############################################

print("="*80)
print("ADVANCED CAUSAL ML WITH EconML")
print("="*80)

# Load data
df = pd.read_csv('./results/gpt-4.1-mini-exclusive-treatment-run2/all_experiment_results_20250821_054934.csv')

# Parse persona JSON
def parse_persona(persona_str):
    try:
        persona_dict = json.loads(persona_str.replace("'", '"'))
        return pd.Series(persona_dict)
    except:
        return pd.Series()

persona_df = df['persona'].apply(parse_persona)
df = pd.concat([df, persona_df], axis=1)

# Feature engineering
feature_cols = ['age', 'income', 'financial_literacy', 'media_exposure', 
                'risk_attitude', 'expenditure', 'pre_treatment_expectation', 
                'pre_confidence']

# Create dummies for categorical
categorical_cols = ['gender', 'education', 'region', 'urban_rural']
for col in categorical_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())

# Prepare data
X = df[feature_cols].fillna(df[feature_cols].median()).values
y = df['expectation_change'].values

# For simplicity, analyze one treatment vs control
treatment_of_interest = 'full_policy_context'
df_subset = df[df['treatment_group'].isin([treatment_of_interest, 'control'])].copy()
X_subset = df_subset[feature_cols].fillna(df_subset[feature_cols].median()).values
y_subset = df_subset['expectation_change'].values
T_subset = (df_subset['treatment_group'] == treatment_of_interest).astype(int).values

print(f"\nAnalyzing: {treatment_of_interest} vs control")
print(f"Sample size: {len(X_subset)} ({T_subset.sum()} treated, {len(T_subset)-T_subset.sum()} control)")
print(f"Features: {len(feature_cols)}")

#############################################
# 1. LINEAR DML (Chernozhukov et al.)
#############################################

print("\n" + "="*80)
print("1. LINEAR DOUBLE MACHINE LEARNING")
print("="*80)

# First stage models
model_y = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_t = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Linear DML
linear_dml = LinearDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=True,
    linear_first_stages=False,
    cv=5,
    random_state=42
)

linear_dml.fit(Y=y_subset, T=T_subset, X=X_subset)

# Get ATE
ate_dml = linear_dml.ate(X=X_subset)

print(f"\nAverage Treatment Effect (ATE):")
print(f"  Point estimate: {ate_dml:.4f}")

# Get CATE
cate_dml = linear_dml.effect(X=X_subset)
print(f"\nConditional Treatment Effects (CATE):")
print(f"  Mean: {np.mean(cate_dml):.4f}")
print(f"  Std:  {np.std(cate_dml):.4f}")
print(f"  Min:  {np.min(cate_dml):.4f}")
print(f"  Max:  {np.max(cate_dml):.4f}")

# Get coefficients (heterogeneity drivers)
coef = linear_dml.coef_
print(f"\nHeterogeneity Coefficients (Top 5):")
feature_names = feature_cols
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coef.flatten()
}).sort_values('coefficient', key=abs, ascending=False)

print(coef_df.head(10).to_string(index=False))

#############################################
# 2. CAUSAL FOREST DML
#############################################

print("\n" + "="*80)
print("2. CAUSAL FOREST DML (Athey & Wager)")
print("="*80)

cf_dml = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100, random_state=42),
    model_t=GradientBoostingRegressor(n_estimators=100, random_state=42),
    discrete_treatment=True,
    n_estimators=100,
    min_samples_leaf=20,
    max_depth=10,
    cv=5,
    random_state=42,
    inference=False  # Turn off for speed
)

cf_dml.fit(Y=y_subset, T=T_subset, X=X_subset)

# Get ATE
ate_cf = cf_dml.ate(X=X_subset)
print(f"\nAverage Treatment Effect (ATE): {ate_cf:.4f}")

# Get CATE
cate_cf = cf_dml.effect(X=X_subset)
print(f"\nConditional Treatment Effects (CATE):")
print(f"  Mean: {np.mean(cate_cf):.4f}")
print(f"  Std:  {np.std(cate_cf):.4f}")
print(f"  Min:  {np.min(cate_cf):.4f}")
print(f"  Max:  {np.max(cate_cf):.4f}")

# Feature importance
feature_importance = cf_dml.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance for Heterogeneity (Top 10):")
print(importance_df.head(10).to_string(index=False))

#############################################
# 3. DOUBLY ROBUST LEARNER
#############################################

print("\n" + "="*80)
print("3. DOUBLY ROBUST LEARNER")
print("="*80)

dr_learner = DRLearner(
    model_propensity=LogisticRegressionCV(cv=5, random_state=42),
    model_regression=GradientBoostingRegressor(n_estimators=100, random_state=42),
    model_final=GradientBoostingRegressor(n_estimators=100, random_state=42),
    cv=5,
    random_state=42
)

dr_learner.fit(Y=y_subset, T=T_subset, X=X_subset)

# Get ATE
ate_dr = dr_learner.ate(X=X_subset)
print(f"\nAverage Treatment Effect (ATE): {ate_dr:.4f}")

# Get CATE
cate_dr = dr_learner.effect(X=X_subset)
print(f"\nConditional Treatment Effects (CATE):")
print(f"  Mean: {np.mean(cate_dr):.4f}")
print(f"  Std:  {np.std(cate_dr):.4f}")
print(f"  Min:  {np.min(cate_dr):.4f}")
print(f"  Max:  {np.max(cate_dr):.4f}")

#############################################
# 4. X-LEARNER (Künzel et al.)
#############################################

print("\n" + "="*80)
print("4. X-LEARNER")
print("="*80)

x_learner = XLearner(
    models=GradientBoostingRegressor(n_estimators=100, random_state=42),
    propensity_model=LogisticRegressionCV(cv=5, random_state=42),
    cate_models=GradientBoostingRegressor(n_estimators=100, random_state=42)
)

x_learner.fit(Y=y_subset, T=T_subset, X=X_subset)

# Get CATE
cate_x = x_learner.effect(X=X_subset)
print(f"\nConditional Treatment Effects (CATE):")
print(f"  Mean: {np.mean(cate_x):.4f}")
print(f"  Std:  {np.std(cate_x):.4f}")
print(f"  Min:  {np.min(cate_x):.4f}")
print(f"  Max:  {np.max(cate_x):.4f}")

#############################################
# 5. MODEL COMPARISON
#############################################

print("\n" + "="*80)
print("5. MODEL COMPARISON")
print("="*80)

results_summary = pd.DataFrame({
    'Method': ['Linear DML', 'Causal Forest DML', 'DR-Learner', 'X-Learner'],
    'ATE': [ate_dml, ate_cf, ate_dr, np.mean(cate_x)],
    'CATE_Mean': [np.mean(cate_dml), np.mean(cate_cf), np.mean(cate_dr), np.mean(cate_x)],
    'CATE_Std': [np.std(cate_dml), np.std(cate_cf), np.std(cate_dr), np.std(cate_x)],
    'CATE_Min': [np.min(cate_dml), np.min(cate_cf), np.min(cate_dr), np.min(cate_x)],
    'CATE_Max': [np.max(cate_dml), np.max(cate_cf), np.max(cate_dr), np.max(cate_x)]
})

print("\n" + results_summary.to_string(index=False))

#############################################
# 6. POLICY LEARNING - WHO SHOULD GET TREATMENT?
#############################################

print("\n" + "="*80)
print("6. POLICY LEARNING: OPTIMAL TREATMENT ASSIGNMENT")
print("="*80)

# Using Causal Forest CATE predictions
# Recommend treatment for those with largest expected benefit

cate_threshold = np.percentile(cate_cf, 25)  # Bottom quartile (most negative)
recommended_treatment = cate_cf < cate_threshold

print(f"\nPolicy recommendation:")
print(f"  Recommend treatment for {recommended_treatment.sum()} individuals")
print(f"  ({100*recommended_treatment.sum()/len(recommended_treatment):.1f}% of sample)")
print(f"\n  Characteristics of recommended group:")

# Analyze recommended group
df_analysis = df_subset.copy()
df_analysis['recommended'] = recommended_treatment
df_analysis['cate'] = cate_cf

recommended_group = df_analysis[df_analysis['recommended']]
not_recommended = df_analysis[~df_analysis['recommended']]

print(f"    Mean age: {recommended_group['age'].mean():.1f} vs {not_recommended['age'].mean():.1f}")
print(f"    Mean financial literacy: {recommended_group['financial_literacy'].mean():.1f} vs {not_recommended['financial_literacy'].mean():.1f}")
print(f"    Mean media exposure: {recommended_group['media_exposure'].mean():.1f} vs {not_recommended['media_exposure'].mean():.1f}")
print(f"    Mean income: {recommended_group['income'].mean()/1e6:.2f}M vs {not_recommended['income'].mean()/1e6:.2f}M")

# Expected value of policy
naive_policy_value = np.mean(y_subset[T_subset==1]) - np.mean(y_subset[T_subset==0])
optimal_policy_value = np.mean(cate_cf[recommended_treatment])

print(f"\n  Policy value analysis:")
print(f"    Naive policy (treat everyone): {naive_policy_value:.4f}")
print(f"    Optimal policy (treat selected): {optimal_policy_value:.4f}")
print(f"    Improvement: {optimal_policy_value - naive_policy_value:.4f}")

#############################################
# 7. VISUALIZATIONS
#############################################

print("\n" + "="*80)
print("7. GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: CATE comparison across methods
fig, ax = plt.subplots(figsize=(14, 6))

methods = ['Linear DML', 'Causal Forest', 'DR-Learner', 'X-Learner']
cates = [cate_dml.flatten(), cate_cf.flatten(), cate_dr.flatten(), cate_x.flatten()]

positions = np.arange(len(methods))
bp = ax.boxplot(cates, positions=positions, widths=0.6, patch_artist=True,
                showmeans=True, meanline=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='green', linewidth=2, linestyle='--'))

ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xticks(positions)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel('Conditional Treatment Effect', fontsize=12)
ax.set_title('CATE Distribution Across Different Causal ML Methods', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./results/causalml/econml_cate_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: econml_cate_comparison.png")

# Plot 2: Feature importance from Causal Forest
fig, ax = plt.subplots(figsize=(10, 8))
top_features = importance_df.head(15)
ax.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].str.replace('_', ' ').str.title())
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance for Treatment Effect Heterogeneity\n(Causal Forest DML)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('./results/causalml/econml_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: econml_feature_importance.png")

# Plot 3: CATE vs observed treatment effect
fig, ax = plt.subplots(figsize=(10, 8))

# Bin CATE predictions and compute actual treatment effects
df_analysis['cate_bin'] = pd.qcut(df_analysis['cate'], q=10, labels=False, duplicates='drop')

bin_results = []
for bin_id in sorted(df_analysis['cate_bin'].unique()):
    bin_data = df_analysis[df_analysis['cate_bin'] == bin_id]
    
    treated = bin_data[bin_data['treatment_group'] == treatment_of_interest]['expectation_change']
    control = bin_data[bin_data['treatment_group'] == 'control']['expectation_change']
    
    if len(treated) > 0 and len(control) > 0:
        actual_effect = treated.mean() - control.mean()
        predicted_effect = bin_data['cate'].mean()
        
        bin_results.append({
            'bin': bin_id,
            'predicted': predicted_effect,
            'actual': actual_effect,
            'n': len(bin_data)
        })

bin_df = pd.DataFrame(bin_results)

ax.scatter(bin_df['predicted'], bin_df['actual'], s=bin_df['n']*3, alpha=0.6)
ax.plot(bin_df['predicted'], bin_df['predicted'], 'r--', linewidth=2, label='Perfect calibration')

# Add trend line
z = np.polyfit(bin_df['predicted'], bin_df['actual'], 1)
p = np.poly1d(z)
ax.plot(bin_df['predicted'], p(bin_df['predicted']), 'g-', linewidth=2, alpha=0.7, label='Actual fit')

ax.set_xlabel('Predicted CATE (Causal Forest)', fontsize=12)
ax.set_ylabel('Actual Treatment Effect (Binned)', fontsize=12)
ax.set_title('CATE Calibration: Predicted vs Actual Treatment Effects', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results/causalml/econml_cate_calibration.png', dpi=300, bbox_inches='tight')
print("✓ Saved: econml_cate_calibration.png")

# Plot 4: Treatment assignment policy
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
axes[0,0].hist(recommended_group['age'], bins=20, alpha=0.5, label='Recommended', color='green')
axes[0,0].hist(not_recommended['age'], bins=20, alpha=0.5, label='Not recommended', color='red')
axes[0,0].set_xlabel('Age', fontsize=10)
axes[0,0].set_ylabel('Frequency', fontsize=10)
axes[0,0].legend()
axes[0,0].set_title('Age Distribution by Policy Recommendation', fontsize=11, fontweight='bold')

# Financial literacy
axes[0,1].hist(recommended_group['financial_literacy'], bins=10, alpha=0.5, label='Recommended', color='green')
axes[0,1].hist(not_recommended['financial_literacy'], bins=10, alpha=0.5, label='Not recommended', color='red')
axes[0,1].set_xlabel('Financial Literacy', fontsize=10)
axes[0,1].set_ylabel('Frequency', fontsize=10)
axes[0,1].legend()
axes[0,1].set_title('Financial Literacy by Policy Recommendation', fontsize=11, fontweight='bold')

# Income
axes[1,0].hist(recommended_group['income']/1e6, bins=20, alpha=0.5, label='Recommended', color='green')
axes[1,0].hist(not_recommended['income']/1e6, bins=20, alpha=0.5, label='Not recommended', color='red')
axes[1,0].set_xlabel('Income (Millions IDR)', fontsize=10)
axes[1,0].set_ylabel('Frequency', fontsize=10)
axes[1,0].legend()
axes[1,0].set_title('Income by Policy Recommendation', fontsize=11, fontweight='bold')

# CATE distribution
axes[1,1].hist(recommended_group['cate'], bins=20, alpha=0.5, label='Recommended', color='green')
axes[1,1].hist(not_recommended['cate'], bins=20, alpha=0.5, label='Not recommended', color='red')
axes[1,1].axvline(x=cate_threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
axes[1,1].set_xlabel('Estimated CATE', fontsize=10)
axes[1,1].set_ylabel('Frequency', fontsize=10)
axes[1,1].legend()
axes[1,1].set_title('CATE Distribution by Policy Recommendation', fontsize=11, fontweight='bold')

plt.suptitle('Optimal Treatment Assignment Policy Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/causalml/econml_policy_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: econml_policy_analysis.png")

print("\n" + "="*80)
print("ADVANCED CAUSAL ML ANALYSIS COMPLETE")
print("="*80)
print("\nKey files generated:")
print("  • econml_cate_comparison.png - Comparison across methods")
print("  • econml_feature_importance.png - Heterogeneity drivers")
print("  • econml_cate_calibration.png - Model validation")
print("  • econml_policy_analysis.png - Optimal assignment policy")
