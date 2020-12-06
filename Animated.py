# Imports.
import plotly.express as px
import numpy as np 
import pandas as pd 


# Define the files being read from
US_counties_file = '../covid-19-data/us-counties.csv'
US_population_file = 'popData.csv'

# Read the files as dataframes
COVID = pd.read_csv(US_counties_file)
pop = pd.read_csv(US_population_file)

# Remove the work county from county names.
pop.CTYNAME = pop.CTYNAME.replace({' County':''}, regex=True)

# Define a new column to hold the percantage of each state's population that live in each county.
pop['pop_pct'] = 0

# Fill in that percentage.
for state in pop.STNAME.unique():
    
    statepop = pop.POPESTIMATE2018[pop.STNAME == state].to_numpy()[0]
    pop.pop_pct[pop.STNAME == state] = pop.POPESTIMATE2018[pop.STNAME == state]/statepop
  

# I don't need the population of the state itself now so we ditch it.
pop = pop[pop.pop_pct != 1]

# for date in COVID.date.unique():
#     # Take the state data and spread it around each county in that state.
#     for state in COVID.state[(COVID.county == 'Unknown') & (COVID.date == date)]:
        
#         cases = COVID.cases[(COVID.county == 'Unknown') & (COVID.date == date) & (COVID.state == state)].to_numpy() *pop.pop_pct[pop.STNAME == state].to_numpy()
#         deaths = COVID.deaths[(COVID.county == 'Unknown') & (COVID.date == date) & (COVID.state == state)].to_numpy() *pop.pop_pct[pop.STNAME == state].to_numpy()
        
#         df = pd.DataFrame(data = {'date':date, 'county': pop[pop.STNAME == state].CTYNAME, 'state':state, 'fips':pop[pop.STNAME == state].FIPS, 'cases':cases, 'deaths':deaths})
        
#         COVID = COVID.append(df)


COVID = pd.read_csv('COVID_filled.csv')