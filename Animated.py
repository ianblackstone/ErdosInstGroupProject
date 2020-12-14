# Imports.
# import plotly.express as px
import numpy as np 
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt


# # Define the files being read from
US_counties_file = '../covid-19-data/us-counties.csv'
US_population_file = 'popData.csv'


# # Read the files as dataframes
# COVID = pd.read_csv(US_counties_file)
pop = pd.read_csv(US_population_file)

# # Remove the work county from county names.
pop.CTYNAME = pop.CTYNAME.replace({' County':''}, regex=True)

# # Define a new column to hold the percantage of each state's population that live in each county.
# pop['pop_pct'] = 0

# # Fill in that percentage.
# for state in pop.STNAME.unique():
    
#     statepop = pop.POPESTIMATE2018[pop.STNAME == state].to_numpy()[0]
#     pop.pop_pct[pop.STNAME == state] = pop.POPESTIMATE2018[pop.STNAME == state]/statepop
  

# # I don't need the population of the state itself now so we ditch it.
# pop = pop[pop.pop_pct != 1]

# for date in COVID.date.unique():
#     # Take the state data and spread it around each county in that state.
#     for state in COVID.state[(COVID.county == 'Unknown') & (COVID.date == date)]:
        
#         cases = COVID.cases[(COVID.county == 'Unknown') & (COVID.date == date) & (COVID.state == state)].to_numpy() *pop.pop_pct[pop.STNAME == state].to_numpy()
#         deaths = COVID.deaths[(COVID.county == 'Unknown') & (COVID.date == date) & (COVID.state == state)].to_numpy() *pop.pop_pct[pop.STNAME == state].to_numpy()
        
#         df = pd.DataFrame(data = {'date':date, 'county': pop[pop.STNAME == state].CTYNAME, 'state':state, 'fips':pop[pop.STNAME == state].FIPS, 'cases':cases, 'deaths':deaths})
        
#         COVID = COVID.append(df)

# Write to file, this takes a long time to do so we don't want to run it every time.
# COVID.to_csv('COVID_filled.csv', index = False)

# Read the data.
COVID = pd.read_csv('COVID_by_County/COVID_by_County.csv')

# COVID = COVID.groupby(['date','fips','state','county']).sum().reset_index()

COVID = pd.merge(COVID, pop[['FIPS','POPESTIMATE2018']], left_on = 'countyFIPS', right_on = 'FIPS', how = 'left').drop(columns='FIPS')

COVID = COVID.rename(columns={'countyFIPS':'fips'})

COVID['cases_per_100k'] = COVID.cases/COVID.POPESTIMATE2018 * 100000

sd = 'us_county.shp'

mapdf = gpd.read_file(sd)

mapdf['fips']=mapdf['fips'].astype('float')

vmin = 0
vmax = COVID.cases_per_100k.max()

dates = COVID.Dates.unique()

for date in dates[70:]:
    rdf=COVID[COVID['Dates'].str.contains(date)]
    
    merged=pd.merge(mapdf,rdf,how='inner',on='fips')
    
    fil = ['Northern Mariana Islands', 'Puerto Rico', 'Virgin Islands', 'Hawaii', 'Alaska']
    merged = merged[~merged.state.isin(fil)]
        
    # create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(14,6))
    
    # add a title and annotation
    ax.set_title('Cases of Covid ' + date, fontdict={'fontsize': '25', 'fontweight' : '3'})
    
    # create map
    merged.plot(column='cases_per_100k', cmap='OrRd', linewidth=1, ax=ax, edgecolor='.5')
    
    # remove the axis
    ax.axis('off')
    
    # Create colorbar legend
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    
    # empty array for the data range
    sm.set_array([])
    
    #add colorbar
    cbar = fig.colorbar(sm)
    fig.savefig('images/' + date + '.png')
    plt.close(fig=None)