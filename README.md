# Synthetic Renewable Energy Data Generation





## Power Curve Scraping

Information is scraped from www.wind-turbine-models.com. Scraping has to be done in a set order

1. Run get_power_curve.py
2. Run get_power_curve_specs.py

You should not have to make any changes at all for the scripts to run.

You get five .csv files containing data on ~400 different turbins as a result, their content is as follows:

1. turbine_power.csv                    --> Powercurves for all ~400 wind turbines
2. turbine_cp_data.csv                  --> Cp data for some few turbines
3. turbine_ct_cata.csv                  --> Ct data for some few turbines
4. turbine_specs.csv                    --> Rotor diameter along with possible hub heights for most turbines 
                                        (values might not always be numeric, or contain values at all)
5. turbine_names.csv