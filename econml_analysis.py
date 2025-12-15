"""
Advanced Causal ML Analysis using EconML
Implements: DML, Causal Forest, DR-Learner, X-Learner
Now includes: SHAP values for interpretability, Policy Trees, and Multi-treatment support.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
import shap
import warnings
import os

warnings.filterwarnings('ignore')

# EconML imports
from econml.dml import LinearDML, CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import XLearner, TLearner

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create results directory if not exists
os.makedirs('./results/causalml', exist_ok=True)

#############################################
# LOAD AND PREPARE DATA
#############################################

print("="*80)
print("ADVANCED CAUSAL ML WITH EconML (Enhanced)")
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
                'risk_attitude', 'expenditure']

# Create dummies for categorical
categorical_cols = ['gender', 'education', 'region', 'urban_rural']
for col in categorical_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())

# Clean column names for SHAP
feature_cols = [c.replace(' ', '_').replace('-', '_') for c in feature_cols]
df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]

# Identify treatments
if 'treatment_group' in df.columns:
    all_treatments = [t for t in df['treatment_group'].unique() if t != 'control']
else:
    # Fallback if treatment_group column is missing but dummies exist
    all_treatments = [c.replace('treat_', '') for c in df.columns if c.startswith('treat_') and c != 'treat_control']

print(f"Found treatments: {all_treatments}")
print(f"Features: {len(feature_cols)}")

#############################################
# ANALYSIS FUNCTION
#############################################

def analyze_treatment(treatment_name, df, feature_cols):
    print(f"\n{'='*60}")
    print(f"ANALYZING: {treatment_name} vs CONTROL")
    print(f"{'='*60}")
    
    # Prepare data for this treatment
    df_subset = df[df['treatment_group'].isin([treatment_name, 'control'])].copy()
    
    # Handle missing values
    X = df_subset[feature_cols].fillna(df_subset[feature_cols].median())
    y = df_subset['expectation_change'].values
    T = (df_subset['treatment_group'] == treatment_name).astype(int).values
    
    X_val = X.values
    
    print(f"Sample size: {len(X)} ({T.sum()} treated, {len(T)-T.sum()} control)")
    
    # 1. CAUSAL FOREST DML (Best for heterogeneity)
    print("\nTraining Causal Forest DML...")
    cf_dml = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        discrete_treatment=True,
        n_estimators=200,
        min_samples_leaf=10,
        max_depth=None,
        cv=5,
        random_state=42
    )
    
    cf_dml.fit(Y=y, T=T, X=X_val)
    
    # Calculate effects
    ate = cf_dml.ate(X=X_val)
    cate = cf_dml.effect(X=X_val)
    
    print(f"ATE: {ate:.4f}")
    print(f"CATE Mean: {np.mean(cate):.4f} (Std: {np.std(cate):.4f})")
    
    # 2. SHAP VALUES FOR INTERPRETABILITY
    print("\nCalculating SHAP values...")
    try:
        # Explain the CATE predictions directly using a surrogate model
        # (Direct SHAP on EconML models can be unstable due to complex structure)
        
        # Train a surrogate XGBoost/RF to predict CATE from X
        surrogate = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        surrogate.fit(X, cate)
        
        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(X)
        
        # Plot SHAP Summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'SHAP Values for Heterogeneity: {treatment_name}\n(Feature impact on Treatment Effect)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'./results/causalml/shap_summary_{treatment_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved SHAP summary plot")
        
    except Exception as e:
        print(f"Error calculating SHAP: {e}")

    # 3. POLICY TREE (Interpretable Rules)
    print("\nGenerating Policy Tree...")
    try:
        # Train a simple decision tree to predict CATE
        # This gives us interpretable rules for who has high/low treatment effects
        policy_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20)
        policy_tree.fit(X, cate)
        
        # Text representation
        rules = export_text(policy_tree, feature_names=feature_cols)
        print("\nPolicy Rules (Predicting CATE):")
        print(rules)
        
        # Visual representation
        plt.figure(figsize=(16, 8))
        plot_tree(policy_tree, feature_names=feature_cols, filled=True, rounded=True, fontsize=10)
        plt.title(f'Policy Tree: Who responds to {treatment_name}?', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'./results/causalml/policy_tree_{treatment_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Policy Tree plot")
        
    except Exception as e:
        print(f"Error generating Policy Tree: {e}")

    return {
        'treatment': treatment_name,
        'ate': ate,
        'cate_mean': np.mean(cate),
        'cate_std': np.std(cate)
    }

#############################################
# MAIN LOOP
#############################################

results = []

for treatment in all_treatments:
    res = analyze_treatment(treatment, df, feature_cols)
    results.append(res)

# Summary table
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('./results/causalml/heterogeneity_summary.csv', index=False)
print("\nAnalysis complete. Results saved to ./results/causalml/")
