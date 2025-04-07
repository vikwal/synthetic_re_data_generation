# Synthetic Renewable Energy Data Generation





## Power Curve Scraping

Information is scraped from www.wind-turbine-models.com. Scraping has to be done in a set order

1. Run windmodel_csv.py
2. Run get_power_curve_specs.py

You should not have to make any changes at all for the scripts to run.

You get four .csv files containing data on ~400 different turbins as a result, their content is as follows:

1. turbine_data.csv                 --> Powercurves for all ~400 wind turbines
2. turbine_cp_data_processed.csv    --> Cp data for some few turbines
3. turbine_ct_cata_processed.csv    --> Ct data for some few turbines
4. turbine_specs_processed.csv      --> Rotor diameter along with possible hub heights for most turbines 
                                        (hub_height might not always be numeric, or contain values at all)