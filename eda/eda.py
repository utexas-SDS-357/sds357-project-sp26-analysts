import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("community_area_data.csv")
print(df.head())


# Histogram of citation_pct
plt.figure(figsize=(8, 5))
plt.hist(df['citation_pct'], bins=20, edgecolor='black') # bin size that best shows dist
plt.xlabel('Citation rate (proportion of stops with citation)')
plt.ylabel('Number of community areas')
plt.title('Distribution of Citation Rates Across Community Areas')
plt.tight_layout()
plt.show()

# Scatter: n_stops vs citation_pct
plt.figure(figsize=(8, 5))
plt.scatter(df['n_stops'], df['citation_pct'], alpha=0.7, edgecolors='k', linewidths=0.5) # customization for visual appeal/clarity
plt.xlabel('Number of stops (n_stops)')
plt.ylabel('Citation rate (citation_pct)')
plt.title('Citation Rate vs. Stop Volume by Community Area')
plt.tight_layout()
plt.show()

