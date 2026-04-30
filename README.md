[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MFzEnxem)

# Understanding Neighborhood-Level Factors Associated with Traffic Stop Citations in Chicago

This project analyzes how neighborhood-level socioeconomic factors are associated with the likelihood of receiving a citation during traffic stops in Chicago. The analysis focuses on **interpretation rather than prediction**, using a grouped binomial logistic regression model at the community area level.

---

## Data Sources

This project uses three publicly available datasets:

1. **Stanford Open Policing Project (Chicago)**  
   https://openpolicing.stanford.edu/data/  
   - Contains individual-level traffic stop records  

2. **Chicago Community Areas (Shapefile)**  
   https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-Map/cauq-8yn6d  
   - Used to map traffic stops to community areas  

3. **Chicago Health Atlas**  
   https://chicagohealthatlas.org/download  
   - Provides neighborhood-level socioeconomic and public health variables  

Download all datasets before running any notebooks.

---

## Data Wrangling

Navigate to the `data_wrangling/` folder and run notebooks in the following order:

1. **`data_wrangling.ipynb`**  
   - Cleans and merges all raw datasets  
   - Performs spatial joins to assign traffic stops to community areas  

2. **`community_area_grouping.ipynb`**  
   - Aggregates the merged dataset to the **community area level**  
   - Creates the final dataset used for modeling  

---

## Exploratory Data Analysis (EDA)

Navigate to the `eda/` folder:

- **`eda.ipynb`**  
  - Performs initial exploratory data analysis on the merged dataset  
  - Examines distributions, missingness, and relationships between variables  
  - Provides context for understanding the structure of the data  

- **`correlation_map.ipynb`**  
  - Generates correlation matrices for community-level variables  
  - Identifies multicollinearity between predictors  
  - Guides selection of variables used in the final model  

---

## Modeling

Navigate to the `Modeling/` folder:

- **`grouped_binomial_logistic.ipynb`**  
  - Fits a grouped binomial logistic regression model using community-level data  
  - Evaluates model fit using weighted mean absolute error (MAE)  
  - Produces coefficient estimates and standardized coefficient plots for interpretation  

---

## Reproducing Results

To reproduce the full analysis:

1. Download all datasets listed above  
2. Clone this repository  
3. Run notebooks in the following order:
   - `data_wrangling/data_wrangling.ipynb`
   - `data_wrangling/community_area_grouping.ipynb`
   - `eda/correlation_map.ipynb`
   - `Modeling/grouped_binomial_logistic.ipynb`

---

## Notes

- The final model operates at the **community area level**, not individual stops  
- The focus is on **associations between socioeconomic factors and citation likelihood**, not causal inference  
- Results may vary slightly depending on preprocessing choices  

---
