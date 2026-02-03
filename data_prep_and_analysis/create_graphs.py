import matplotlib
matplotlib.use('Agg')

import pandas
import matplotlib.pyplot as plt
import seaborn as sns



ds = pandas.read_csv('data/training_data_encoded.csv')

plt.style.use('seaborn-v0_8-whitegrid')

##target dist. histogram
plt.figure(figsize=(8, 5))
plt.hist(ds['outcome'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Target Variable', fontsize=14)
plt.axvline(ds['outcome'].mean(), color='red', linestyle='--', label=f'Mean: {ds["outcome"].mean():.2f}')
plt.axvline(ds['outcome'].median(), color='orange', linestyle='--', label=f'Median: {ds["outcome"].median():.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('graphs/outcome_distribution_postencode.png', dpi=150, bbox_inches='tight')
plt.close()

##scatter depth vs outcome
plt.figure(figsize=(8, 6))
plt.scatter(ds['depth'], ds['outcome'], alpha=0.3, s=10, color='steelblue')
plt.xlabel('Depth', fontsize=12)
plt.ylabel('Outcome', fontsize=12)
plt.title(f'Depth vs Outcome (r = {ds["depth"].corr(ds["outcome"]):.3f})', fontsize=14)
plt.tight_layout()
plt.savefig('graphs/depth_vs_outcome_postencode.png', dpi=150, bbox_inches='tight')
plt.close()


##correlation bar chart
numerical_cols = ds.select_dtypes(include=['float64', 'int64']).columns
correlations = ds[numerical_cols].corr()['outcome'].drop('outcome').sort_values(key=abs, ascending=False)
#select top 10
top_corr = correlations.head(10)


plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in top_corr.values]
plt.barh(top_corr.index, top_corr.values, color=colors, edgecolor='black')
plt.xlabel('Correlation with Outcome', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 10 Features by Correlation with Outcome', fontsize=14)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('graphs/correlation_bar_chart_postencode.png', dpi=150, bbox_inches='tight')
plt.close()