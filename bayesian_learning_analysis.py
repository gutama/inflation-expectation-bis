import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.iolib.summary2 import summary_col
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
print("Loading data...")
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
df_clean = df.dropna(subset=['pre_expectation', 'post_expectation', 
                               'pre_conf', 'post_conf', 'treatment_group'])

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
print("SUMMARY STATISTICS")
print("="*80)

# Overall statistics
summary_stats = df_clean[['pre_expectation', 'post_expectation', 'exp_change',
                           'pre_conf', 'post_conf', 'conf_change']].describe()
print("\n", summary_stats)

# By treatment group
print("\n\nBy Treatment Group:")
print("-" * 80)
treatment_summary = df_clean.groupby([col for col in df_clean.columns if col.startswith('treat_')])[
    ['pre_expectation', 'post_expectation', 'exp_change', 'pre_conf', 'post_conf', 'conf_change']
].agg(['mean', 'std', 'count'])

# Get original treatment names
df_temp = df[['treatment_group']].copy()
df_temp = pd.get_dummies(df_temp, columns=['treatment_group'], prefix='treat')
if 'treat_control' in df_temp.columns:
    df_temp = df_temp.drop('treat_control', axis=1)

for col in treatment_cols:
    treat_name = col.replace('treat_', '')
    subset = df_clean[df_clean[col] == 1]
    print(f"\n{treat_name.upper()} (n={len(subset)}):")
    print(subset[['pre_expectation', 'post_expectation', 'exp_change', 
                  'pre_conf', 'post_conf', 'conf_change']].describe().loc[['mean', 'std']])

# Control group
if 'treatment_group' in df.columns:
    control = df[df['treatment_group'] == 'control']
    print(f"\nCONTROL (n={len(control)}):")
    print(control[['pre_treatment_expectation', 'post_treatment_expectation', 
                   'expectation_change', 'pre_confidence', 'post_confidence']].describe().loc[['mean', 'std']])

print("\n" + "="*80)
print("CONFIDENCE ANALYSIS: Testing for Perceived Signal Precision")
print("="*80)

# Test if confidence changed within each treatment
from scipy import stats

print("\n1. Within-Treatment Confidence Changes (Paired t-tests):")
print("-" * 80)

all_treatments = df['treatment_group'].unique()
for treatment in all_treatments:
    subset = df_clean[df_clean.get(f'treat_{treatment}', 
                                     df['treatment_group'] == treatment) == (1 if f'treat_{treatment}' in df_clean.columns else True)]
    if len(subset) == 0:
        subset = df[df['treatment_group'] == treatment].dropna(subset=['pre_confidence', 'post_confidence'])
    
    if len(subset) > 0:
        pre = pd.to_numeric(subset.get('pre_conf', subset.get('pre_confidence')), errors='coerce')
        post = pd.to_numeric(subset.get('post_conf', subset.get('post_confidence')), errors='coerce')
        
        # Remove any remaining NaNs
        valid = ~(pre.isna() | post.isna())
        pre = pre[valid]
        post = post[valid]
        
        if len(pre) > 1:
            t_stat, p_val = stats.ttest_rel(post, pre)
            mean_change = (post - pre).mean()
            
            print(f"\n{treatment}:")
            print(f"  Mean confidence change: {mean_change:.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''}")
            print(f"  n = {len(pre)}")

# Test if treatment groups differ in confidence change from control
print("\n\n2. Between-Treatment Confidence Changes (vs Control):")
print("-" * 80)

if 'treatment_group' in df.columns:
    control_data = df[df['treatment_group'] == 'control'].copy()
    control_conf_change = pd.to_numeric(control_data['post_confidence'], errors='coerce') - \
                          pd.to_numeric(control_data['pre_confidence'], errors='coerce')
    control_conf_change = control_conf_change.dropna()
    
    for treatment in all_treatments:
        if treatment != 'control':
            treat_data = df[df['treatment_group'] == treatment].copy()
            treat_conf_change = pd.to_numeric(treat_data['post_confidence'], errors='coerce') - \
                               pd.to_numeric(treat_data['pre_confidence'], errors='coerce')
            treat_conf_change = treat_conf_change.dropna()
            
            if len(treat_conf_change) > 0 and len(control_conf_change) > 0:
                t_stat, p_val = stats.ttest_ind(treat_conf_change, control_conf_change)
                diff = treat_conf_change.mean() - control_conf_change.mean()
                
                print(f"\n{treatment} vs control:")
                print(f"  Difference in confidence change: {diff:.4f}")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''}")

print("\n" + "="*80)
print("BAYESIAN LEARNING TESTS")
print("="*80)

# Prepare regression data
reg_data = df_clean.copy()

# Create interaction terms: prior * treatment
for treat_col in treatment_cols:
    reg_data[f'{treat_col}_x_prior'] = reg_data[treat_col] * reg_data['pre_expectation']

print("\n1. BASELINE MODEL (No Controls)")
print("-" * 80)

# Model 1: Posterior ~ Prior + Treatments
formula1 = 'post_expectation ~ pre_expectation'
for treat_col in treatment_cols:
    formula1 += f' + {treat_col}'

model1 = ols(formula1, data=reg_data).fit()
print(model1.summary())

print("\n2. BAYESIAN LEARNING MODEL (Prior × Treatment Interactions)")
print("-" * 80)

# Model 2: Posterior ~ Prior + Treatments + Prior*Treatments
formula2 = 'post_expectation ~ pre_expectation'
for treat_col in treatment_cols:
    formula2 += f' + {treat_col} + {treat_col}_x_prior'

model2 = ols(formula2, data=reg_data).fit()
print(model2.summary())

# Interpretation of interaction coefficients
print("\n\nInterpretation of Prior × Treatment Interactions:")
print("-" * 60)
print("Positive coefficient: Treatment increases weight on prior (more anchoring)")
print("Negative coefficient: Treatment decreases weight on prior (more updating)")
print("Near-zero coefficient: Treatment doesn't affect Bayesian updating\n")

for treat_col in treatment_cols:
    interact_coef = model2.params.get(f'{treat_col}_x_prior', None)
    interact_pval = model2.pvalues.get(f'{treat_col}_x_prior', None)
    
    if interact_coef is not None:
        treatment_name = treat_col.replace('treat_', '').upper()
        sig = '***' if interact_pval < 0.01 else '**' if interact_pval < 0.05 else '*' if interact_pval < 0.1 else ''
        print(f"{treatment_name}:")
        print(f"  Coefficient: {interact_coef:.4f} {sig}")
        print(f"  p-value: {interact_pval:.4f}")
        
        if abs(interact_coef) < 0.05:
            print(f"  → Minimal effect on Bayesian updating (near-zero)")
        elif interact_coef > 0:
            print(f"  → Increases anchoring to prior beliefs")
        else:
            print(f"  → Reduces anchoring to prior beliefs (more updating)")
        print()

# Check if we have demographic controls
demo_vars = []
for var in ['age', 'financial_literacy', 'media_exposure', 'risk_attitude']:
    if var in reg_data.columns and reg_data[var].notna().sum() > 100:
        reg_data[var] = pd.to_numeric(reg_data[var], errors='coerce')
        demo_vars.append(var)

# Gender dummy
if 'gender' in reg_data.columns:
    reg_data['female'] = (reg_data['gender'] == 'Female').astype(int)
    demo_vars.append('female')

# Urban dummy
if 'urban_rural' in reg_data.columns:
    reg_data['urban'] = (reg_data['urban_rural'] == 'Urban').astype(int)
    demo_vars.append('urban')

# Education dummies - clean column names
if 'education' in reg_data.columns:
    edu_dummies = pd.get_dummies(reg_data['education'], prefix='edu', drop_first=True)
    # Clean column names to avoid formula parsing issues
    edu_dummies.columns = [col.replace("'", "").replace('"', '').replace(' ', '_') for col in edu_dummies.columns]
    reg_data = pd.concat([reg_data, edu_dummies], axis=1)
    demo_vars.extend([col for col in edu_dummies.columns])

reg_data_demo = reg_data.dropna(subset=['pre_expectation', 'post_expectation'] + demo_vars)

if len(demo_vars) > 0 and len(reg_data_demo) > 100:
    print("\n3. WITH DEMOGRAPHIC CONTROLS")
    print("-" * 80)
    
    # Model 3: Add demographic controls
    formula3 = 'post_expectation ~ pre_expectation'
    for treat_col in treatment_cols:
        formula3 += f' + {treat_col}'
    for demo_var in demo_vars:
        formula3 += f' + {demo_var}'
    
    model3 = ols(formula3, data=reg_data_demo).fit()
    print(model3.summary())
    
    print("\n4. BAYESIAN LEARNING WITH CONTROLS (Prior × Treatment + Demographics)")
    print("-" * 80)
    
    # Model 4: Interactions + demographics
    formula4 = 'post_expectation ~ pre_expectation'
    for treat_col in treatment_cols:
        formula4 += f' + {treat_col} + {treat_col}_x_prior'
    for demo_var in demo_vars:
        formula4 += f' + {demo_var}'
    
    # Recreate interactions for demo data
    for treat_col in treatment_cols:
        reg_data_demo[f'{treat_col}_x_prior'] = reg_data_demo[treat_col] * reg_data_demo['pre_expectation']
    
    model4 = ols(formula4, data=reg_data_demo).fit()
    print(model4.summary())
    
    print("\n\nComparison of Models:")
    print("="*80)
    comparison = summary_col([model1, model2, model3, model4],
                            model_names=['Baseline', 'With Interactions', 
                                        'With Controls', 'Full Model'],
                            stars=True,
                            info_dict={'N': lambda x: f"{int(x.nobs)}",
                                      'R-squared': lambda x: f"{x.rsquared:.4f}"})
    print(comparison)
else:
    print("\n\nComparison of Models:")
    print("="*80)
    comparison = summary_col([model1, model2],
                            model_names=['Baseline', 'With Interactions'],
                            stars=True,
                            info_dict={'N': lambda x: f"{int(x.nobs)}",
                                      'R-squared': lambda x: f"{x.rsquared:.4f}"})
    print(comparison)

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create comprehensive figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Scatter plots: Posterior vs Prior by Treatment
treatment_colors = {'control': 'gray', 
                    'current_inflation': 'blue',
                    'inflation_target': 'green',
                    'policy_rate_decision': 'orange',
                    'media_narrative': 'red',
                    'full_policy_context': 'purple'}

for idx, treatment in enumerate(all_treatments):
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    if 'treatment_group' in df.columns:
        treat_data = df[df['treatment_group'] == treatment].copy()
        pre = pd.to_numeric(treat_data['pre_treatment_expectation'], errors='coerce')
        post = pd.to_numeric(treat_data['post_treatment_expectation'], errors='coerce')
    else:
        treat_col = f'treat_{treatment}'
        if treat_col in df_clean.columns:
            treat_data = df_clean[df_clean[treat_col] == 1]
            pre = treat_data['pre_expectation']
            post = treat_data['post_expectation']
        else:
            continue
    
    # Remove NaNs
    valid = ~(pre.isna() | post.isna())
    pre = pre[valid]
    post = post[valid]
    
    if len(pre) > 0:
        color = treatment_colors.get(treatment, 'black')
        
        # Scatter plot with jitter
        ax.scatter(pre + np.random.normal(0, 0.05, len(pre)), 
                  post + np.random.normal(0, 0.05, len(post)),
                  alpha=0.4, s=30, color=color)
        
        # Add 45-degree line (no updating)
        min_val = min(pre.min(), post.min())
        max_val = max(pre.max(), post.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1, label='No updating')
        
        # Fit regression line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(pre, post)
        line_x = np.array([min_val, max_val])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color=color, linewidth=2, alpha=0.8, 
               label=f'Fit: β={slope:.2f}')
        
        ax.set_xlabel('Prior Expectation (%)', fontsize=10)
        ax.set_ylabel('Posterior Expectation (%)', fontsize=10)
        ax.set_title(f'{treatment.replace("_", " ").title()}\n(n={len(pre)})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

# Save scatter plots
plt.suptitle('Bayesian Learning Test: Posterior vs Prior Expectations by Treatment', 
            fontsize=16, fontweight='bold', y=0.995)
scatter_file = 'results/bayesian/bayesian_learning_scatter.png'
plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
print(f"\nSaved scatter plots to: {scatter_file}")
plt.close()

# 2. Confidence Analysis Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Confidence change distribution by treatment
ax = axes[0, 0]
conf_changes_by_treat = []
labels = []
for treatment in all_treatments:
    if 'treatment_group' in df.columns:
        treat_data = df[df['treatment_group'] == treatment].copy()
        conf_change = pd.to_numeric(treat_data['post_confidence'], errors='coerce') - \
                     pd.to_numeric(treat_data['pre_confidence'], errors='coerce')
    else:
        treat_col = f'treat_{treatment}'
        if treat_col in df_clean.columns:
            treat_data = df_clean[df_clean[treat_col] == 1]
            conf_change = treat_data['conf_change']
        else:
            continue
    
    conf_change = conf_change.dropna()
    if len(conf_change) > 0:
        conf_changes_by_treat.append(conf_change)
        labels.append(treatment.replace('_', '\n'))

bp = ax.boxplot(conf_changes_by_treat, labels=labels, patch_artist=True)
for patch, treatment in zip(bp['boxes'], all_treatments):
    patch.set_facecolor(treatment_colors.get(treatment, 'lightblue'))
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No change')
ax.set_ylabel('Change in Confidence', fontsize=12, fontweight='bold')
ax.set_title('A. Distribution of Confidence Changes by Treatment', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

# Panel B: Pre vs Post confidence scatter
ax = axes[0, 1]
for treatment in all_treatments:
    if 'treatment_group' in df.columns:
        treat_data = df[df['treatment_group'] == treatment].copy()
        pre_conf = pd.to_numeric(treat_data['pre_confidence'], errors='coerce')
        post_conf = pd.to_numeric(treat_data['post_confidence'], errors='coerce')
    else:
        treat_col = f'treat_{treatment}'
        if treat_col in df_clean.columns:
            treat_data = df_clean[df_clean[treat_col] == 1]
            pre_conf = treat_data['pre_conf']
            post_conf = treat_data['post_conf']
        else:
            continue
    
    valid = ~(pre_conf.isna() | post_conf.isna())
    pre_conf = pre_conf[valid]
    post_conf = post_conf[valid]
    
    if len(pre_conf) > 0:
        color = treatment_colors.get(treatment, 'black')
        ax.scatter(pre_conf, post_conf, alpha=0.3, s=20, color=color, 
                  label=treatment.replace('_', ' ').title())

ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('Pre-Treatment Confidence', fontsize=12, fontweight='bold')
ax.set_ylabel('Post-Treatment Confidence', fontsize=12, fontweight='bold')
ax.set_title('B. Pre vs Post Confidence (All Treatments)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel C: Mean confidence by treatment
ax = axes[1, 0]
treatment_names = []
pre_means = []
post_means = []

for treatment in all_treatments:
    if 'treatment_group' in df.columns:
        treat_data = df[df['treatment_group'] == treatment].copy()
        pre_conf = pd.to_numeric(treat_data['pre_confidence'], errors='coerce')
        post_conf = pd.to_numeric(treat_data['post_confidence'], errors='coerce')
    else:
        treat_col = f'treat_{treatment}'
        if treat_col in df_clean.columns:
            treat_data = df_clean[df_clean[treat_col] == 1]
            pre_conf = treat_data['pre_conf']
            post_conf = treat_data['post_conf']
        else:
            continue
    
    if len(pre_conf.dropna()) > 0:
        treatment_names.append(treatment.replace('_', '\n'))
        pre_means.append(pre_conf.mean())
        post_means.append(post_conf.mean())

x = np.arange(len(treatment_names))
width = 0.35

bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-Treatment', alpha=0.7, color='steelblue')
bars2 = ax.bar(x + width/2, post_means, width, label='Post-Treatment', alpha=0.7, color='coral')

ax.set_ylabel('Mean Confidence', fontsize=12, fontweight='bold')
ax.set_title('C. Mean Confidence Levels by Treatment', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(treatment_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Confidence change vs expectation change
ax = axes[1, 1]
for treatment in all_treatments:
    if 'treatment_group' in df.columns:
        treat_data = df[df['treatment_group'] == treatment].copy()
        exp_change = pd.to_numeric(treat_data['expectation_change'], errors='coerce')
        conf_change = pd.to_numeric(treat_data['post_confidence'], errors='coerce') - \
                     pd.to_numeric(treat_data['pre_confidence'], errors='coerce')
    else:
        treat_col = f'treat_{treatment}'
        if treat_col in df_clean.columns:
            treat_data = df_clean[df_clean[treat_col] == 1]
            exp_change = treat_data['exp_change']
            conf_change = treat_data['conf_change']
        else:
            continue
    
    valid = ~(exp_change.isna() | conf_change.isna())
    exp_change = exp_change[valid]
    conf_change = conf_change[valid]
    
    if len(exp_change) > 0:
        color = treatment_colors.get(treatment, 'black')
        ax.scatter(exp_change, conf_change, alpha=0.3, s=20, color=color,
                  label=treatment.replace('_', ' ').title())

ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Expectation Change (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Confidence Change', fontsize=12, fontweight='bold')
ax.set_title('D. Confidence Change vs Expectation Update', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3)

plt.suptitle('Signal Precision Analysis: Did Treatments Increase Confidence?', 
            fontsize=16, fontweight='bold')
conf_file = 'results/bayesian/confidence_precision_analysis.png'
plt.savefig(conf_file, dpi=300, bbox_inches='tight')
print(f"Saved confidence analysis to: {conf_file}")
plt.close()

# 3. Bayesian Learning Coefficients Plot
fig, ax = plt.subplots(figsize=(12, 8))

# Extract interaction coefficients
interaction_results = []
for treat_col in treatment_cols:
    treatment_name = treat_col.replace('treat_', '')
    coef = model2.params.get(f'{treat_col}_x_prior', None)
    se = model2.bse.get(f'{treat_col}_x_prior', None)
    pval = model2.pvalues.get(f'{treat_col}_x_prior', None)
    
    if coef is not None:
        interaction_results.append({
            'treatment': treatment_name.replace('_', ' ').title(),
            'coefficient': coef,
            'se': se,
            'ci_lower': coef - 1.96*se,
            'ci_upper': coef + 1.96*se,
            'pval': pval
        })

interaction_df = pd.DataFrame(interaction_results)
interaction_df = interaction_df.sort_values('coefficient')

# Plot
y_pos = np.arange(len(interaction_df))
colors = ['red' if p < 0.05 else 'gray' for p in interaction_df['pval']]

ax.errorbar(interaction_df['coefficient'], y_pos, 
           xerr=[interaction_df['coefficient'] - interaction_df['ci_lower'],
                 interaction_df['ci_upper'] - interaction_df['coefficient']],
           fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2)

for i, (coef, color) in enumerate(zip(interaction_df['coefficient'], colors)):
    ax.plot(coef, i, 'o', markersize=10, color=color)

ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='No interaction')
ax.set_yticks(y_pos)
ax.set_yticklabels(interaction_df['treatment'])
ax.set_xlabel('Coefficient on Prior × Treatment Interaction', fontsize=12, fontweight='bold')
ax.set_title('Bayesian Learning: Do Treatments Affect Weight on Priors?\n(Positive = More Anchoring, Negative = More Updating)', 
            fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.legend()

# Add significance stars
for i, pval in enumerate(interaction_df['pval']):
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    if sig:
        ax.text(interaction_df.iloc[i]['ci_upper'] + 0.01, i, sig, 
               fontsize=12, fontweight='bold', va='center')

coef_file = 'results/bayesian/bayesian_learning_coefficients.png'
plt.tight_layout()
plt.savefig(coef_file, dpi=300, bbox_inches='tight')
print(f"Saved coefficient plot to: {coef_file}")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Findings:")
print("-" * 80)
print("\n1. CONFIDENCE/PRECISION:")
print("   - Check if any treatment significantly increased confidence")
print("   - Near-zero changes suggest signals not perceived as precise")
print("\n2. BAYESIAN LEARNING:")
print("   - Prior coefficient shows baseline anchoring")
print("   - Treatment × Prior interactions show if treatments affect updating")
print("   - Near-zero interactions = no effect on Bayesian learning")
print("\n3. INTERPRETATION:")
print("   - Flat confidence + minimal interactions → Persistent ambiguity")
print("   - Not classic Bayesian learning from high-precision signals")
print("\nAll results saved to results/bayesian/")
print("="*80)
