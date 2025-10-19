import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Data from analysis
treatments = ['Current\nInflation', 'Inflation\nTarget', 'Full Policy\nContext', 
              'Policy Rate\nDecision', 'Media\nNarrative']

# Anchoring reduction (from interaction coefficients)
anchoring_reduction = [54.1, 71.4, 65.9, 14.5, 8.7]

# Confidence change (mean change)
confidence_change = [0.012, -0.018, 0.064, 0.166, 0.159]

# Significance
significant = [False, False, True, True, True]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: The Paradox - Scatter plot
colors = ['red' if sig else 'gray' for sig in significant]
sizes = [200 if sig else 100 for sig in significant]

ax1.scatter(anchoring_reduction, confidence_change, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=2)

# Add treatment labels
for i, (x, y, txt) in enumerate(zip(anchoring_reduction, confidence_change, treatments)):
    offset = 0.015 if i < 3 else -0.025
    ax1.annotate(txt.replace('\n', ' '), (x, y), xytext=(5, offset*100), 
                textcoords='offset points', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow' if significant[i] else 'lightgray', alpha=0.7))

# Add quadrant lines
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
ax1.axvline(x=30, color='black', linestyle='--', linewidth=1, alpha=0.3)

# Add quadrant labels
ax1.text(10, 0.17, 'Confidence\nWithout\nUpdating', fontsize=10, alpha=0.5, ha='center', style='italic')
ax1.text(65, 0.17, 'Bayesian\nIdeal:\nBoth', fontsize=10, alpha=0.5, ha='center', style='italic', color='green')
ax1.text(10, -0.04, 'Neither', fontsize=10, alpha=0.5, ha='center', style='italic')
ax1.text(65, -0.04, '⚠️ THE PARADOX:\nUpdating\nWithout\nConfidence', fontsize=11, alpha=0.7, ha='center', 
         style='italic', color='darkred', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='pink', alpha=0.3))

ax1.set_xlabel('Reduction in Anchoring (%)\n← Less Updating | More Updating →', fontsize=13, fontweight='bold')
ax1.set_ylabel('Change in Confidence (0-10 scale)\n← Less Confident | More Confident →', fontsize=13, fontweight='bold')
ax1.set_title('The Updating-Confidence Paradox', fontsize=15, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 80)
ax1.set_ylim(-0.05, 0.19)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.7, label='Significant (p<0.05)'),
                   Patch(facecolor='gray', alpha=0.7, label='Not Significant')]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=11)

# Panel 2: Bar chart comparison
x = np.arange(len(treatments))
width = 0.35

# Normalize for visual comparison
anchoring_norm = [a/100 for a in anchoring_reduction]  # Scale to 0-1
confidence_norm = [(c + 0.05) / 0.25 for c in confidence_change]  # Scale to 0-1

bars1 = ax2.bar(x - width/2, anchoring_norm, width, label='Updating Magnitude\n(% Reduction in Anchoring)', 
                alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, confidence_norm, width, label='Confidence Change\n(Normalized)', 
                alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    
    ax2.text(bar1.get_x() + bar1.get_width()/2., height1,
            f'{anchoring_reduction[i]:.0f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    significance_marker = '***' if significant[i] else ''
    ax2.text(bar2.get_x() + bar2.get_width()/2., height2,
            f'{confidence_change[i]:.3f}{significance_marker}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Normalized Magnitude (0-1 scale)', fontsize=12, fontweight='bold')
ax2.set_title('Updating vs. Confidence: Direct Comparison', fontsize=15, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(treatments, fontsize=11)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.1)

# Add annotation explaining the pattern
ax2.annotate('Technical info:\nHigh updating, low confidence',
            xy=(0.5, 0.5), xytext=(1.3, 0.85),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkred'),
            fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='pink', alpha=0.5))

ax2.annotate('Narrative info:\nLow updating, high confidence',
            xy=(3.5, 0.75), xytext=(2.2, 0.95),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'),
            fontsize=11, fontweight='bold', color='darkblue',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))

plt.suptitle('Key Finding: Updating and Confidence Are Decoupled', 
            fontsize=17, fontweight='bold', y=0.98)

# Add footnote
fig.text(0.5, 0.01, 
        'Note: Classic Bayesian learning predicts positive correlation between updating and confidence.\n' +
        'Observed pattern suggests persistent model uncertainty (ambiguity) rather than parameter uncertainty (risk).\n' +
        '*** p<0.01 for confidence change vs control',
        ha='center', fontsize=10, style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('results/bayesian/updating_confidence_paradox.png', dpi=300, bbox_inches='tight')
print("Saved paradox visualization to: results/bayesian/updating_confidence_paradox.png")
plt.close()

# Create one more summary figure showing effective prior weights
fig, ax = plt.subplots(figsize=(12, 7))

treatments_full = ['Control', 'Current\nInflation', 'Inflation\nTarget', 'Full Policy\nContext',
                   'Policy Rate\nDecision', 'Media\nNarrative']
baseline_weight = 0.758
effective_weights = [0.758, 0.348, 0.217, 0.258, 0.648, 0.692]
colors_weights = ['gray', 'steelblue', 'steelblue', 'steelblue', 'coral', 'coral']

bars = ax.bar(range(len(treatments_full)), effective_weights, color=colors_weights, 
              alpha=0.7, edgecolor='black', linewidth=2)

# Add baseline reference line
ax.axhline(y=baseline_weight, color='red', linestyle='--', linewidth=2, 
          label=f'Baseline (Control) = {baseline_weight}', alpha=0.7)

# Add value labels
for i, (bar, weight) in enumerate(zip(bars, effective_weights)):
    height = bar.get_height()
    reduction = ((baseline_weight - weight) / baseline_weight * 100) if i > 0 else 0
    
    label_text = f'{weight:.3f}'
    if i > 0:
        label_text += f'\n({reduction:.0f}% ↓)'
    
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           label_text,
           ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Effective Weight on Prior Expectation', fontsize=13, fontweight='bold')
ax.set_xlabel('Treatment Condition', fontsize=13, fontweight='bold')
ax.set_title('How Much Do Treatments Reduce Anchoring to Priors?', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(range(len(treatments_full)))
ax.set_xticklabels(treatments_full, fontsize=11)
ax.set_ylim(0, 0.9)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Add interpretation zones
ax.axhspan(0, 0.3, alpha=0.1, color='green', label='Strong updating zone')
ax.axhspan(0.6, 0.9, alpha=0.1, color='red', label='Weak updating zone')

# Add annotations
ax.annotate('Technical info:\nStrong reduction in anchoring',
           xy=(2, 0.25), xytext=(2.5, 0.15),
           arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'),
           fontsize=11, fontweight='bold', color='darkgreen',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))

ax.annotate('Narrative info:\nMinimal reduction',
           xy=(4.5, 0.67), xytext=(3.5, 0.5),
           arrowprops=dict(arrowstyle='->', lw=2, color='darkred'),
           fontsize=11, fontweight='bold', color='darkred',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='pink', alpha=0.5))

fig.text(0.5, 0.01,
        'Note: Lower values = less weight on prior = more updating in response to treatment.\n' +
        'All treatment × prior interactions significant at p<0.01 except Media Narrative (p=0.14)',
        ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.98])
plt.savefig('results/bayesian/effective_prior_weights.png', dpi=300, bbox_inches='tight')
print("Saved prior weights visualization to: results/bayesian/effective_prior_weights.png")
plt.close()

print("\n✓ All visualizations created successfully!")
print("\nFiles created:")
print("  1. updating_confidence_paradox.png - Main finding visualization")
print("  2. effective_prior_weights.png - Treatment effects on anchoring")
