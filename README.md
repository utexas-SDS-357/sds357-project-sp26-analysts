[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MFzEnxem)

## Data

This project requires three datasets:

1. **Stanford Open Policing Project (Chicago):** https://openpolicing.stanford.edu/data/
2. **Chicago Community Areas Shapefile:** https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-Map/cauq-8yn6d
3. **Chicago Health Atlas:** https://chicagohealthatlas.org/download

## Data Wrangling

- `data_wrangling/data_wrangling.ipynb` — Merges and cleans the three raw datasets into the final `traffic_analysis.csv` used for analysis.
- `data_wrangling/grouped_vio.ipynb` — Adds an AI-based violation categorization column (`grouped_vio`) to the dataset for use in modelling. *Need to provide OpenAI API key to run this.
