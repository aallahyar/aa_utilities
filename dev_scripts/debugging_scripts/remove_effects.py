import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from aa_utilities.computation.modeling import remove_effects


# 1) Metadata: 24 samples, 
# batch (A/B) as confounder, 
# condition (Control/Case) as signal
n = 24
np.random.seed(123)
meta = pd.DataFrame({
    'sample_id': [f"S{i:02d}" for i in range(n)],
    'batch': np.random.choice(['A', 'B'], size=n),
    'condition': np.random.choice(['Control', 'Case'], size=n),
}).set_index('sample_id')

batch_B = (meta['batch'] == 'B').astype(float)      # confounder to remove
case = (meta['condition'] == 'Case').astype(float)  # signal to preserve

# 2) Simulate a single gene's expression: base + condition + batch + noise
base = 6.0
cond_effect = 0.5      # we want to keep this
batch_effect = 0.4     # we want to remove this
noise = np.random.normal(0, 0.4, size=n)

gene_expr = (
    base
    + cond_effect * case.values
    + batch_effect * batch_B.values
    + noise
)

expr_df = pd.DataFrame({'GeneX': gene_expr}, index=meta.index)

# 3) Adjust for batch while preserving condition (residualization)
X_full = pd.DataFrame({'Intercept': 1.0, 'Condition': case, 'BatchB': batch_B}, index=meta.index)
X_signal = X_full[['Intercept', 'Condition']]

y = expr_df['GeneX'].values
m_full = sm.OLS(y, X_full.values).fit()
m_sig = sm.OLS(y, X_signal.values).fit()

batch_component = m_full.predict(X_full.values) - m_sig.predict(X_signal.values)
gene_expr_adj = y - batch_component

# adj_df = pd.DataFrame({'GeneX_adj': gene_expr_adj}, index=meta.index)
adj_df = remove_effects(
    dataframe=pd.concat([expr_df, meta], axis=1),
    response='GeneX',
    covs_all=['condition', 'batch'],
    covs_remove=['batch'],
    verbose=True
).response_adjusted.to_frame(name='GeneX_adj')

# 4) Plot before vs after correction (by condition and by batch)
def plot_before_after(meta, original, adjusted):
    d = meta.copy()
    d['GeneX'] = original
    d['GeneX_adj'] = adjusted
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    # Side-by-side boxplots by condition
    m_cond = d.melt(id_vars=['condition'], value_vars=['GeneX','GeneX_adj'],
                    var_name='state', value_name='expr')
    print(m_cond)
    sns.boxplot(data=m_cond, x='condition', y='expr', hue='state', ax=axes[0])
    sns.stripplot(data=m_cond, x='condition', y='expr', hue='state', dodge=True, palette='dark:black', alpha=0.5, ax=axes[0])
    axes[0].set_title('GeneX: Original vs Adjusted (by Condition)')
    axes[0].legend(title='')

    # Side-by-side boxplots by batch (to demonstrate batch removal)
    m_batch = d.melt(id_vars=['batch'], value_vars=['GeneX','GeneX_adj'],
                     var_name='state', value_name='expr')
    sns.boxplot(data=m_batch, x='batch', y='expr', hue='state', ax=axes[1])
    sns.stripplot(data=m_batch, x='batch', y='expr', hue='state', dodge=True, palette='dark:black', alpha=0.5, ax=axes[1])
    axes[1].set_title('GeneX: Original vs Adjusted (by Batch)')
    axes[1].legend(title='')

    plt.tight_layout()
    plt.show()

plot_before_after(meta, expr_df['GeneX'].values, adj_df['GeneX_adj'].values)

# 5) Quick numeric summary
diff_by_batch_original = expr_df.groupby(meta['batch']).mean().rename(columns={'GeneX':'Original'})
diff_by_batch_adjusted = adj_df.groupby(meta['batch']).mean().rename(columns={'GeneX_adj':'Adjusted'})
summary = diff_by_batch_original.join(diff_by_batch_adjusted)
print("Mean expression by batch (Original vs Adjusted):")
print(summary)