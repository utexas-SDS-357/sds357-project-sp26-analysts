import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

df = pd.read_csv("traffic_analysis.csv")

# replacing nan values of citation_issued with False
df["citation_issued"] = df["citation_issued"].astype("boolean")   # nullable boolean dtype
df["citation_issued"] = df["citation_issued"].fillna(False)       # fill NA with False
# print(df['citation_issued'].head) # confirming it worked
# print(set(df['citation_issued'])) #confirming there are both T and F


# print(set(df["violation"]))
print(len(set(df["violation"]))) #1107 unique violation codes

#plt.hist(df["subject_age"])
#plt.show()

# na_count = df.isna().sum()
# print(na_count)

# print(set(df['citation_issued']))


# missing a ton of values in citation_issued, so replace NA with False
# cor between citation_issued and search_conducted is -0.05
# ~0.6% searched, ~31.8% cited -> search frequency is so small that it would have such a huge class imbalance
# so, switching outcome to just citations bc adding search makes minimal difference
# assuming that missing values in citation_issued represent falses since only trues are logged. this is a limitation but assuming so that we can use it as our outcome

# penalized regression to not overfit? e.g. lasso, ridge
# later for modeling, use train/test/predict splits

# subset data to rows with officer demographics, check if % citations is roughly the same, and if so can use that. maybe within -6% to +20%

# check that outcome keeps highest step

# print(set(df['outcome']))
#check proportions of outcomes against binary individual outcome proportions