import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.iolib.summary2 import summary_col
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*80)
print("BAYESIAN LEARNING ANALYSIS WITH WEBER ET AL. (2025) METHODOLOGY")
print("Tell Me Something I Don't Already Know: Learning in Low- and High-Inflation Settings")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv('results/gpt-4.1-mini-exclusive-treatment-run2/all_experiment_results_20250821_054934.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Parse persona column if needed
if 'persona' in df.columns and isinstance(df['persona'].iloc[0], str):
    print("\nParsing persona column...")
    import ast
    df['persona_dict'] = df['persona'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
    
    # Extract demographic variables
    for key in ['age', 'gender', 'education', 'financial_literacy', 'media_exposure', 
                'risk_attitude', 'urban_rural', 'income', 'expenditure']:
        if key not in df.columns:
            df[key] = df['persona_dict'].apply(lambda x: x.get(key, np.nan))

# Clean and prepare data
print("\nData cleaning...")
df['pre_expectation'] = pd.to_numeric(df['pre_treatment_expectation'], errors='coerce')
df['post_expectation'] = pd.to_numeric(df['post_treatment_expectation'], errors='coerce')
df['pre_conf'] = pd.to_numeric(df['pre_confidence'], errors='coerce')
df['post_conf'] = pd.to_numeric(df['post_confidence'], errors='coerce')
df['exp_change'] = df['post_expectation'] - df['pre_expectation']
df['conf_change'] = df['post_conf'] - df['pre_conf']

# Remove missing values
df_clean = df.dropna(subset=['pre_expectation', 'post_expectation', 'treatment_group'])

print(f"Clean dataset shape: {df_clean.shape}")
print(f"\nTreatment groups: {df_clean['treatment_group'].unique()}")
print(f"\nTreatment counts:\n{df_clean['treatment_group'].value_counts()}")

# Create treatment dummies
df_clean = pd.get_dummies(df_clean, columns=['treatment_group'], prefix='treat', drop_first=False)
treatment_cols = [col for col in df_clean.columns if col.startswith('treat_')]
print(f"\nTreatment dummy variables: {treatment_cols}")

# Set control as baseline (drop it)
if 'treat_control' in df_clean.columns:
    df_clean = df_clean.drop('treat_control', axis=1)
    treatment_cols.remove('treat_control')

print("\n" + "="*80)
print("WEBER ET AL. (2025) SPECIFICATION: EQUATION (1)")
print("="*80)
print("\nModel: posterior_i = α + β × prior_i + δ × I_i + γ × I_i × prior_i + error_i")
print("\nWhere:")
print("  - I_i = treatment indicator")
print("  - β = slope for control group (baseline weight on priors)")
print("  - γ = change in slope for treatment group")
print("  - Scaled treatment effect = γ/β (key metric)")
print("\nInterpretation:")
print("  - γ/β ≈ 0: No learning from treatment (fully attentive)")
print("  - γ/β ≈ -1: Full learning from treatment (fully inattentive)")
print("  - |γ/β| measures degree of inattention")

# Prepare regression data
reg_data = df_clean.copy()

# Create interaction terms: prior * treatment
for treat_col in treatment_cols:
    reg_data[f'{treat_col}_x_prior'] = reg_data[treat_col] * reg_data['pre_expectation']

print("\n" + "="*80)
print("1. BASELINE MODEL: Control Group Only")
print("="*80)
print("\nEstimating β (baseline weight on priors)...")

# Control group analysis
control_data = reg_data[[not any(reg_data[col].iloc[i] == 1 for col in treatment_cols) 
                         for i in range(len(reg_data))]].copy()

if len(control_data) > 0:
    control_model = ols('post_expectation ~ pre_expectation', data=control_data).fit()
    beta_control = control_model.params['pre_expectation']
    se_beta = control_model.bse['pre_expectation']
    
    print(f"\nControl Group Results:")
    print(f"  β (slope on prior) = {beta_control:.4f} (SE = {se_beta:.4f})")
    print(f"  R² = {control_model.rsquared:.4f}")
    print(f"  n = {len(control_data)}")
    print("\nInterpretation:")
    if beta_control > 0.9:
        print("  → High anchoring on priors (β close to 1)")
    elif beta_control > 0.5:
        print("  → Moderate anchoring on priors")
    else:
        print("  → Low anchoring on priors")
else:
    print("Warning: No control group found!")
    beta_control = 1.0

print("\n" + "="*80)
print("2. TREATMENT EFFECTS MODEL (Weber et al. Specification)")
print("="*80)

# Full model with interactions
formula = 'post_expectation ~ pre_expectation'
for treat_col in treatment_cols:
    formula += f' + {treat_col} + {treat_col}_x_prior'

print(f"\nFormula: {formula}")
print("\nEstimating with OLS...")
model_ols = ols(formula, data=reg_data).fit()

print("\n" + "-"*80)
print("OLS Results:")
print("-"*80)
print(model_ols.summary())

# Also estimate with robust regression (Huber, as in Weber et al.)
print("\n" + "-"*80)
print("Robust Regression Results (Huber M-estimator):")
print("-"*80)
print("(Used in Weber et al. to reduce influence of outliers)")

model_robust = RLM.from_formula(formula, data=reg_data, M=sm.robust.norms.HuberT()).fit()
print(model_robust.summary())

print("\n" + "="*80)
print("3. SCALED TREATMENT EFFECTS (γ/β) - KEY WEBER METRIC")
print("="*80)

# Calculate scaled effects for each treatment
scaled_results = []

for treat_col in treatment_cols:
    treatment_name = treat_col.replace('treat_', '')
    
    # Get coefficients
    gamma = model_ols.params.get(f'{treat_col}_x_prior', np.nan)
    beta = model_ols.params.get('pre_expectation', beta_control)
    delta = model_ols.params.get(treat_col, np.nan)
    
    # Standard errors
    se_gamma = model_ols.bse.get(f'{treat_col}_x_prior', np.nan)
    se_beta = model_ols.bse.get('pre_expectation', se_beta if 'se_beta' in locals() else np.nan)
    se_delta = model_ols.bse.get(treat_col, np.nan)
    
    # P-values
    pval_gamma = model_ols.pvalues.get(f'{treat_col}_x_prior', np.nan)
    pval_delta = model_ols.pvalues.get(treat_col, np.nan)
    
    # Scaled effect: γ/β
    scaled_effect = gamma / beta if beta != 0 else np.nan
    
    # Approximate SE using delta method: SE(γ/β) ≈ |γ/β| × sqrt((SE_γ/γ)² + (SE_β/β)²)
    if not np.isnan(scaled_effect) and gamma != 0 and beta != 0:
        se_scaled = abs(scaled_effect) * np.sqrt((se_gamma/gamma)**2 + (se_beta/beta)**2)
        ci_lower = scaled_effect - 1.96 * se_scaled
        ci_upper = scaled_effect + 1.96 * se_scaled
    else:
        se_scaled = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
    
    scaled_results.append({
        'treatment': treatment_name,
        'beta': beta,
        'gamma': gamma,
        'delta': delta,
        'scaled_effect': scaled_effect,
        'se_scaled': se_scaled,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'pval_gamma': pval_gamma,
        'pval_delta': pval_delta,
        'n': len(reg_data[reg_data[treat_col] == 1])
    })

results_df = pd.DataFrame(scaled_results)

print("\n" + "-"*80)
print("SCALED TREATMENT EFFECTS (γ/β)")
print("-"*80)
print("\nThis table shows the KEY METRIC from Weber et al. (2025):")
print("  γ/β = How much flatter is the prior-posterior slope for treated agents?")
print("\n")
print(results_df.to_string(index=False))

print("\n\nINTERPRETATION GUIDE:")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"\n{row['treatment'].upper()}:")
    print(f"  γ/β = {row['scaled_effect']:.3f}")
    
    if abs(row['scaled_effect']) < 0.1:
        interpretation = "FULLY ATTENTIVE - Treatment had minimal effect"
        mechanism = "Agents already knew this information (extensive margin)"
    elif abs(row['scaled_effect']) < 0.3:
        interpretation = "HIGHLY ATTENTIVE - Small treatment effect"
        mechanism = "Most agents somewhat informed before treatment"
    elif abs(row['scaled_effect']) < 0.6:
        interpretation = "MODERATELY INATTENTIVE - Moderate treatment effect"
        mechanism = "Substantial learning from treatment (intensive margin)"
    else:
        interpretation = "HIGHLY INATTENTIVE - Large treatment effect"
        mechanism = "Agents placed high weight on new information"
    
    print(f"  → {interpretation}")
    print(f"  → {mechanism}")
    
    if not np.isnan(row['pval_gamma']):
        sig = '***' if row['pval_gamma'] < 0.01 else '**' if row['pval_gamma'] < 0.05 else '*' if row['pval_gamma'] < 0.1 else 'not significant'
        print(f"  → Statistical significance: {sig}")

print("\n" + "="*80)
print("4. DECOMPOSITION: Intensive vs Extensive Margins")
print("="*80)
print("\nFollowing Weber et al. Proposition 2:")
print("  γ/β = - [Var(π|S_i) / (Var(π|S_i) + σ²_νp)] × 1{S_p ∉ S_i}")
print("\nThis decomposes into:")
print("  1. INTENSIVE MARGIN: Kalman gain [Var(π|S_i) / (Var(π|S_i) + σ²_νp)]")
print("     - How strongly do uninformed agents respond?")
print("  2. EXTENSIVE MARGIN: 1{S_p ∉ S_i}")
print("     - Share of agents who didn't already know the signal")

print("\n\nAPPROXIMATE EXTENSIVE MARGIN (Share Already Informed):")
print("-" * 80)
print("Method: Count agents with zero expectation change in each treatment\n")

for treat_col in treatment_cols:
    treatment_name = treat_col.replace('treat_', '')
    treat_data = reg_data[reg_data[treat_col] == 1]
    
    n_total = len(treat_data)
    n_no_change = len(treat_data[abs(treat_data['exp_change']) < 0.01])  # Threshold: <0.01%
    n_small_change = len(treat_data[abs(treat_data['exp_change']) < 0.5])  # Threshold: <0.5%
    
    pct_no_change = 100 * n_no_change / n_total if n_total > 0 else 0
    pct_small_change = 100 * n_small_change / n_total if n_total > 0 else 0
    
    print(f"{treatment_name}:")
    print(f"  No change (|Δ| < 0.01%): {n_no_change}/{n_total} ({pct_no_change:.1f}%)")
    print(f"  Small change (|Δ| < 0.5%): {n_small_change}/{n_total} ({pct_small_change:.1f}%)")
    print(f"  → Estimated share already informed: ~{pct_no_change:.1f}%")
    print()

print("\n" + "="*80)
print("5. SIGNAL PRECISION ANALYSIS (Confidence Changes)")
print("="*80)
print("\nWeber et al. test whether treatments increase confidence")
print("(indicating signal perceived as precise/informative)\n")

all_treatments = df['treatment_group'].unique()

for treatment in all_treatments:
    subset = df[df['treatment_group'] == treatment].copy()
    
    pre = pd.to_numeric(subset['pre_confidence'], errors='coerce')
    post = pd.to_numeric(subset['post_confidence'], errors='coerce')
    
    valid = ~(pre.isna() | post.isna())
    pre = pre[valid]
    post = post[valid]
    
    if len(pre) > 1:
        t_stat, p_val = stats.ttest_rel(post, pre)
        mean_change = (post - pre).mean()
        
        print(f"{treatment}:")
        print(f"  Mean confidence change: {mean_change:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''}")
        
        if abs(mean_change) < 0.1:
            print(f"  → No significant confidence change (signal not perceived as precise)")
        elif mean_change > 0:
            print(f"  → Confidence increased (signal perceived as informative)")
        else:
            print(f"  → Confidence decreased (signal increased uncertainty)")
        print()

print("\n" + "="*80)
print("6. THEORETICAL MECHANISMS (Weber et al. Proposition 4)")
print("="*80)
print("\nWhy might |γ/β| vary across treatments?")
print("\n1. UNCERTAINTY (σ²_π): Higher inflation volatility → more attention")
print("   - Agents process more information when stakes are higher")
print("   - Would predict: |γ/β| decreases with inflation variance")
print("\n2. PERSISTENCE (ρ): More persistent inflation → more attention")
print("   - Current inflation more informative about future")
print("   - Would predict: |γ/β| decreases with inflation persistence")
print("\n3. CREDIBILITY (σ²_νp): Less trust in signal → less response")
print("   - Agents discount noisy/unreliable signals")
print("   - Would predict: |γ/β| decreases with perceived noise")
print("\n4. COST OF INFORMATION (φ or ω): Lower cost → more attention")
print("   - Easier access to information → more people already know")
print("   - Would predict: |γ/β| decreases with information supply")

print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

import os
os.makedirs('results/bayesian_weber', exist_ok=True)

# 1. Scaled Treatment Effects Plot
fig, ax = plt.subplots(figsize=(12, 8))

results_plot = results_df.sort_values('scaled_effect')
y_pos = np.arange(len(results_plot))

# Plot with error bars
colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' 
          for p in results_plot['pval_gamma']]

for i, row in enumerate(results_plot.itertuples()):
    # Error bar
    if not np.isnan(row.ci_lower):
        ax.plot([row.ci_lower, row.ci_upper], [i, i], 'k-', linewidth=2, alpha=0.5)
        ax.plot([row.ci_lower, row.ci_lower], [i-0.1, i+0.1], 'k-', linewidth=2)
        ax.plot([row.ci_upper, row.ci_upper], [i-0.1, i+0.1], 'k-', linewidth=2)
    
    # Point estimate
    ax.plot(row.scaled_effect, i, 'o', markersize=12, color=colors[i])

# Reference lines
ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, 
          label='No effect (fully attentive)')
ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, alpha=0.5, 
          label='Full effect (fully inattentive)')

ax.set_yticks(y_pos)
ax.set_yticklabels([t.replace('_', ' ').title() for t in results_plot['treatment']])
ax.set_xlabel('Scaled Treatment Effect (γ/β)', fontsize=13, fontweight='bold')
ax.set_title('Weber et al. (2025) Metric: How Much Do Treatments Affect Learning?\n' + 
            'Negative values = Treatment reduced weight on priors (induced learning)',
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(fontsize=11, loc='lower right')

# Add significance stars
for i, row in enumerate(results_plot.itertuples()):
    if not np.isnan(row.pval_gamma):
        sig = '***' if row.pval_gamma < 0.01 else '**' if row.pval_gamma < 0.05 else '*' if row.pval_gamma < 0.1 else ''
        if sig:
            x_pos = row.ci_upper if not np.isnan(row.ci_upper) else row.scaled_effect
            ax.text(x_pos + 0.02, i, sig, fontsize=14, fontweight='bold', va='center')

plt.tight_layout()
plt.savefig('results/bayesian_weber/scaled_treatment_effects.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/bayesian_weber/scaled_treatment_effects.png")
plt.close()

# 2. Prior-Posterior Plots by Treatment
n_treatments = len(treatment_cols) + 1  # +1 for control
n_cols = 3
n_rows = int(np.ceil(n_treatments / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
axes = axes.flatten() if n_treatments > 1 else [axes]

# Control group
if len(control_data) > 0:
    ax = axes[0]
    ax.scatter(control_data['pre_expectation'], control_data['post_expectation'], 
              alpha=0.4, s=30, color='gray')
    
    # Regression line
    x_range = np.linspace(control_data['pre_expectation'].min(), 
                         control_data['pre_expectation'].max(), 100)
    y_pred = control_model.params['Intercept'] + control_model.params['pre_expectation'] * x_range
    ax.plot(x_range, y_pred, 'r-', linewidth=2, label=f'β = {beta_control:.3f}')
    
    # 45-degree line
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='No learning (β=1)')
    
    ax.set_xlabel('Prior Expectation', fontsize=11, fontweight='bold')
    ax.set_ylabel('Posterior Expectation', fontsize=11, fontweight='bold')
    ax.set_title('CONTROL GROUP', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Treatment groups
for idx, treat_col in enumerate(treatment_cols):
    ax = axes[idx + 1]
    treatment_name = treat_col.replace('treat_', '')
    treat_data = reg_data[reg_data[treat_col] == 1]
    
    ax.scatter(treat_data['pre_expectation'], treat_data['post_expectation'], 
              alpha=0.4, s=30)
    
    # Calculate actual slope for this treatment
    beta_treat = beta_control + results_df[results_df['treatment'] == treatment_name]['gamma'].values[0]
    scaled = results_df[results_df['treatment'] == treatment_name]['scaled_effect'].values[0]
    
    # Regression line
    if len(treat_data) > 0:
        x_range = np.linspace(treat_data['pre_expectation'].min(), 
                             treat_data['pre_expectation'].max(), 100)
        # Predict using full model
        y_pred = model_ols.params['Intercept']
        y_pred += model_ols.params.get(treat_col, 0)
        y_pred += (beta_control + results_df[results_df['treatment'] == treatment_name]['gamma'].values[0]) * x_range
        ax.plot(x_range, y_pred, 'r-', linewidth=2, 
               label=f'β+γ = {beta_treat:.3f}\nγ/β = {scaled:.3f}')
    
    # 45-degree line
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='β=1')
    
    ax.set_xlabel('Prior Expectation', fontsize=11, fontweight='bold')
    ax.set_ylabel('Posterior Expectation', fontsize=11, fontweight='bold')
    ax.set_title(f'{treatment_name.upper().replace("_", " ")}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Hide extra subplots
for idx in range(n_treatments, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Weber et al. Specification: Prior vs Posterior by Treatment\n' + 
            'Flatter slopes indicate stronger treatment effects',
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/bayesian_weber/prior_posterior_by_treatment.png', dpi=300, bbox_inches='tight')
print("Saved: results/bayesian_weber/prior_posterior_by_treatment.png")
plt.close()

# 3. Comparison to Weber et al. findings
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Scaled effects ordered by magnitude
ax = axes[0, 0]
results_sorted = results_df.sort_values('scaled_effect', ascending=True)
y_pos = np.arange(len(results_sorted))

bars = ax.barh(y_pos, results_sorted['scaled_effect'], 
               color=['darkred' if p < 0.05 else 'coral' if p < 0.1 else 'lightgray' 
                      for p in results_sorted['pval_gamma']])

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=-0.5, color='blue', linestyle='--', alpha=0.5, label='Weber: Low inflation')
ax.axvline(x=-0.2, color='red', linestyle='--', alpha=0.5, label='Weber: High inflation')

ax.set_yticks(y_pos)
ax.set_yticklabels([t.replace('_', '\n') for t in results_sorted['treatment']], fontsize=9)
ax.set_xlabel('γ/β (Scaled Effect)', fontsize=11, fontweight='bold')
ax.set_title('A. Treatment Effects vs Weber et al. Benchmarks', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# Panel B: Gamma coefficients
ax = axes[0, 1]
y_pos = np.arange(len(results_sorted))
colors_gamma = ['darkblue' if p < 0.05 else 'skyblue' if p < 0.1 else 'lightgray' 
                for p in results_sorted['pval_gamma']]

ax.barh(y_pos, results_sorted['gamma'], color=colors_gamma)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels([t.replace('_', '\n') for t in results_sorted['treatment']], fontsize=9)
ax.set_xlabel('γ (Interaction Coefficient)', fontsize=11, fontweight='bold')
ax.set_title('B. Raw Interaction Effects', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Panel C: Extensive margin (approximation)
ax = axes[1, 0]
extensive_data = []
for treat_col in treatment_cols:
    treatment_name = treat_col.replace('treat_', '')
    treat_data = reg_data[reg_data[treat_col] == 1]
    n_total = len(treat_data)
    n_no_change = len(treat_data[abs(treat_data['exp_change']) < 0.01])
    pct_informed = 100 * n_no_change / n_total if n_total > 0 else 0
    extensive_data.append({'treatment': treatment_name, 'pct_informed': pct_informed})

extensive_df = pd.DataFrame(extensive_data).sort_values('pct_informed', ascending=True)
y_pos = np.arange(len(extensive_df))

ax.barh(y_pos, extensive_df['pct_informed'], color='forestgreen', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([t.replace('_', '\n') for t in extensive_df['treatment']], fontsize=9)
ax.set_xlabel('% Already Informed (approx.)', fontsize=11, fontweight='bold')
ax.set_title('C. Extensive Margin: Share Already Knew Info', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Panel D: Confidence changes
ax = axes[1, 1]
conf_changes = []
for treatment in all_treatments:
    subset = df[df['treatment_group'] == treatment].copy()
    pre = pd.to_numeric(subset['pre_confidence'], errors='coerce')
    post = pd.to_numeric(subset['post_confidence'], errors='coerce')
    valid = ~(pre.isna() | post.isna())
    mean_change = (post[valid] - pre[valid]).mean()
    conf_changes.append({'treatment': treatment, 'conf_change': mean_change})

conf_df = pd.DataFrame(conf_changes).sort_values('conf_change', ascending=True)
y_pos = np.arange(len(conf_df))

colors_conf = ['darkgreen' if c > 0.1 else 'lightcoral' if c < -0.1 else 'lightgray' 
               for c in conf_df['conf_change']]

ax.barh(y_pos, conf_df['conf_change'], color=colors_conf, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([t.replace('_', '\n') for t in conf_df['treatment']], fontsize=9)
ax.set_xlabel('Mean Confidence Change', fontsize=11, fontweight='bold')
ax.set_title('D. Signal Precision: Did Confidence Increase?', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('Weber et al. (2025) Framework: Decomposing Treatment Effects',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/bayesian_weber/weber_framework_decomposition.png', dpi=300, bbox_inches='tight')
print("Saved: results/bayesian_weber/weber_framework_decomposition.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Findings (Weber et al. Framework):")
print("-" * 80)

print("\n1. SCALED TREATMENT EFFECTS (γ/β):")
for _, row in results_df.iterrows():
    print(f"   {row['treatment']}: {row['scaled_effect']:.3f} " + 
          f"({'***' if row['pval_gamma'] < 0.01 else '**' if row['pval_gamma'] < 0.05 else '*' if row['pval_gamma'] < 0.1 else 'ns'})")

print("\n2. INTERPRETATION:")
mean_scaled = results_df['scaled_effect'].mean()
if abs(mean_scaled) < 0.2:
    print("   → Overall: HIGH ATTENTION (agents mostly informed)")
    print("   → Similar to Weber et al. high-inflation environment results")
elif abs(mean_scaled) < 0.5:
    print("   → Overall: MODERATE ATTENTION (mixed information)")
elif abs(mean_scaled) < 0.8:
    print("   → Overall: LOW ATTENTION (substantial learning)")
    print("   → Similar to Weber et al. low-inflation environment results")
else:
    print("   → Overall: VERY LOW ATTENTION (strong learning effects)")

print("\n3. THEORETICAL MECHANISMS:")
print("   Based on Weber et al. Proposition 4, variation in |γ/β| could reflect:")
print("   - Differences in prior uncertainty across treatment types")
print("   - Varying credibility of different information sources")
print("   - Heterogeneous costs of acquiring different types of information")

print("\n4. POLICY IMPLICATIONS:")
if abs(mean_scaled) < 0.3:
    print("   → Communication challenge: Agents already well-informed")
    print("   → Need novel/credible information to shift expectations")
else:
    print("   → Communication opportunity: Agents responsive to new information")
    print("   → Information provision can effectively anchor expectations")

print("\nAll results saved to: results/bayesian_weber/")
print("="*80)

print("\n" + "="*80)
print("REFERENCES")
print("="*80)
print("\nWeber, M., Candia, B., Afrouzi, H., Ropele, T., Lluberas, R., Frache, S.,")
print("Meyer, B., Kumar, S., Gorodnichenko, Y., Georgarakos, D., Coibion, O.,")
print("Kenny, G., & Ponce, J. (2025). Tell Me Something I Don't Already Know:")
print("Learning in Low- and High-Inflation Settings. Econometrica, 93(1), 229-264.")
print("\nKey Equations:")
print("  - Equation (1): posterior = α + β×prior + δ×I + γ×I×prior + ε")
print("  - Equation (3): γ/β = -[Var(π|Si)/(Var(π|Si)+σ²νp)] × 1{Sp∉Si}")
print("  - Proposition 4: ∂|γ/β|/∂ρ < 0, ∂|γ/β|/∂σ²νp > 0, etc.")
print("="*80)
