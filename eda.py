import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy import stats

df = pd.read_csv("traffic_analysis.csv")

# replacing nan values of citation_issued with False
df["citation_issued"] = df["citation_issued"].astype("boolean")   # nullable boolean dtype
df["citation_issued"] = df["citation_issued"].fillna(False)
# print(df['citation_issued'].value_counts())
# print(np.mean(df['citation_issued']))


# MAKING VIOLATION FREQUENCY BAR GRAPH
#
counts_violation = Counter(df['violation'])
N = 10 # Take the top N most common
most_common = counts_violation.most_common(N)  # list of (value, count)
# print(most_common)

labels = [k for k, v in most_common]
freqs  = [v for k, v in most_common]
x = np.arange(len(labels))         # numeric positions 0,1,2,...

plt.figure(figsize=(10, 6))
plt.barh(labels, freqs)
plt.xlabel('Frequency', fontsize=15)
plt.ylabel('Violation Type', fontsize=15)
plt.yticks(fontsize=9)
plt.title('10 Most Common Violations', fontsize=16)
plt.tight_layout()
plt.show()
#


# MAKING RACE FREQUENCY BAR GRAPH
#
# counts_race = Counter(df['subject_race'].dropna())
# exclude = ['other', 'unknown']
# for cat in exclude:
#     counts_race.pop(cat, None)
#
# items = sorted(counts_race.items(), key=lambda x: x[1], reverse=True)
# labels_race = [k for k, v in items]
# freqs_race  = [v for k, v in items]
#
# x = np.arange(len(labels_race))
# fig, ax = plt.subplots(figsize=(10,6))
# ax.bar(x, freqs_race, align="center")
# ax.set_xticks(x)
# ax.set_xticklabels(labels_race, ha="center", fontsize=13)
# ax.set_xlabel("Race", fontsize=15)
# ax.set_ylabel("Frequency of Stops", fontsize=15)
# ax.set_title("Stop Frequency by Driver Race", fontsize=16)
# fig.tight_layout()
# plt.show()


# MAKING GROUPED BAR CHART OF CITATION RATE BY RACE
#
# compute citation rate by race
# citation_rates = df.groupby('subject_race')['citation_issued'].mean().sort_values(ascending=False)
# citation_rates = citation_rates.drop(['other', 'unknown'])
#
# # plot grouped bar chart
# plt.figure(figsize=(10, 6))
# citation_rates.plot(kind='bar')
#
# plt.xlabel('Driver Race', fontsize=15)
# plt.ylabel('Citation Rate', fontsize=15)
# plt.title('Citation Rate by Driver Race', fontsize=16)
# plt.xticks(rotation = 0, ha='center', fontsize=13)
# plt.ylim(0, 0.6)
# plt.tight_layout()
# plt.show()

# CREATING PREDICTOR CORRELATIONS AND SIGNIFICANCES FOR STOP COUNT
#
# stop_counts = df.groupby('community_area_id').size().reset_index(name='stop_count')
# area_info = df.groupby('community_area_id')[
#     ['drug_abuse', 'major_crime', 'public_crime', 'violent_crime', 'economic_diversity_index', 'hardship_index', 'median_household_income', 'poverty_rate', 'foreign_born', 'limited_english_proficiency', 'population']
# ].first().reset_index()
# merged = stop_counts.merge(area_info, on='community_area_id')
#
#
# predictors = ['drug_abuse', 'major_crime', 'public_crime', 'violent_crime', 'economic_diversity_index', 'hardship_index', 'median_household_income', 'poverty_rate', 'foreign_born', 'limited_english_proficiency', 'population']
#
#
# results = []
# for col in predictors:
#     slope, intercept, r_value, p_value, std_err = stats.linregress(merged[col], merged['stop_count'])
#     results.append({
#         'Predictor': col,
#         'Slope': round(slope, 4),
#         'R²': round(r_value**2, 4),
#         'P-value': f'{p_value:.4e}'
#     })
#
# results_df = pd.DataFrame(results)
# print(results_df.to_string(index=False))

