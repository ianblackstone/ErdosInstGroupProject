# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

# We get a lot of warnings about slices of copies, these are not meaningful.
import warnings
warnings.filterwarnings('ignore')

# Define our source files
COVID_file = 'COVID_by_County/COVID_by_County.csv'
pop_file = 'popData.csv'
US_county_land_file = 'LND01.csv'
US_county_vote_file = 'president_county_candidate.csv'
mask_use_file = '../covid-19-data/mask-use/mask-use-by-county.csv'
age_file = 'median age.csv'

# Read the source files into dataframes.
COVID_data = pd.read_csv(COVID_file)
pop_data = pd.read_csv(pop_file)
land_data = pd.read_csv(US_county_land_file)
vote_data = pd.read_csv(US_county_vote_file)
mask_use = pd.read_csv(mask_use_file)
age_data = pd.read_csv(age_file)

# Rename some of the columns
COVID_data = COVID_data.rename(columns={'countyFIPS':'fips'})
pop_data = pop_data.rename(columns={'FIPS':'fips', 'Poverty Percent':'Poverty_pct'})
land_data = land_data.rename(columns={'STCOU':'fips','LND010190D':'land'})

# We don't need most of this dataframe so we truncate it down.
land_data = land_data[['fips','land']]

# the vote data is stores as a mixture of row and column data so we jump through hoops to make it usable.
Biden = vote_data[vote_data.candidate == 'Joe Biden']
trump = vote_data[vote_data.candidate == 'Donald Trump']
vote_data = Biden[['state', 'county', 'total_votes']].copy()
vote_data.rename(columns={'total_votes':'Biden'}, inplace=True)
vote_data['Trump'] = trump.total_votes.to_numpy()

# We need some definition that allows us to compare to similar categories.
mask_use['mask_pct'] = mask_use['FREQUENTLY'] + mask_use['ALWAYS']

# merge the vote data into the populationm data, we can't do this on the main dataframe DF because vote data does not use fips.
# pop_data = pd.merge(pop_data, vote_data, left_on = ['CTYNAME', 'STNAME'], right_on = ['county','state'])
pop_data = pd.merge(pop_data, age_data, left_on = 'fips', right_on = 'fips')

# Merge all the datasets into one main dataframe, DF.  Fioltering down when necessary.
DF = pd.merge(COVID_data, pop_data[['fips', 'age', 'POPESTIMATE2018', 'Poverty_pct']], left_on='fips', right_on='fips')
DF = pd.merge(DF,land_data,  left_on='fips', right_on='fips')
DF = pd.merge(DF, mask_use[['COUNTYFP','mask_pct']], left_on = 'fips', right_on = 'COUNTYFP')

# We need to make sure the poverty data is converted from an object type.
DF.Poverty_pct = DF.Poverty_pct.astype('float')

# We don't need NaN entries, they will only cause issues.
DF = DF.dropna()

# Need to fix some counties that have 0 area in the dataset, data manually found with Google.
DF.land[DF['County Name'] == 'Broomfield County and City'] = 33.55
DF.land[DF['County Name'] == 'Yakutat City and Borough'] = 9463
DF.land[DF['County Name'] == 'Denali Borough'] = 12777
DF.land[DF['County Name'] == 'Skagway Municipality'] = 464
DF.land[DF['County Name'] == 'Wrangell City and Borough'] = 3477

# We create new columns containing per capita data
DF['cases_per_cap'] = DF.cases / DF.POPESTIMATE2018
DF['deaths_per_cap'] = DF.deaths / DF.POPESTIMATE2018
DF['pop_density'] = DF.POPESTIMATE2018 / DF.land

# Create a reduced data frame with just one time slice.
RDF = DF[DF['Dates'] == '12/9/2020']

state_score = pd.DataFrame()

state_list = RDF.State.unique()

for st in state_list:

    mask = (RDF['State'] == st)
    
    RDF_train = RDF[~mask]
    RDF_test = RDF[mask]
    
    X_train = RDF_train[['Poverty_pct','pop_density','age','mask_pct']]
    y_cases_train = RDF_train['cases_per_cap']
    y_deaths_train = RDF_train['deaths_per_cap']
    
    X_test = RDF_test[['Poverty_pct','pop_density','age','mask_pct']]
    y_cases_test = RDF_test['cases_per_cap']
    y_deaths_test = RDF_test['deaths_per_cap']
    
    d_range = np.arange(1, 14)
    d_scores = []
    for d in d_range:
        rf = RandomForestRegressor(max_depth = d)
        loss = abs(cross_val_score(rf, X_train, y_deaths_train, cv=5, scoring='neg_mean_squared_error'))
        d_scores.append(loss.mean())
    
    deaths_depth = d_range[d_scores == min(d_scores)][0]
    
    d_scores = []
    for d in d_range:
        rf = RandomForestRegressor(max_depth = d)
        loss = abs(cross_val_score(rf, X_train, y_cases_train, cv=5, scoring='neg_mean_squared_error'))
        d_scores.append(loss.mean())
    
    cases_depth = d_range[d_scores == min(d_scores)][0]
    
    k_range = np.arange(1, 91)
    k_scores = []
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        loss = abs(cross_val_score(knn, X_train, y_cases_train, cv=5, scoring='neg_mean_squared_error'))
        k_scores.append(loss.mean())
    
    cases_nn = k_range[k_scores == min(k_scores)][0]
    
    k_scores = []
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        loss = abs(cross_val_score(knn, X_train, y_deaths_train, cv=5, scoring='neg_mean_squared_error'))
        k_scores.append(loss.mean())
    
    deaths_nn = k_range[k_scores == min(k_scores)][0]  
    
    regr_c = RandomForestRegressor(max_depth = cases_depth)
    regr_c.fit(X_train, y_cases_train)
    
    regr_d = RandomForestRegressor(max_depth = deaths_depth)
    regr_d.fit(X_train, y_deaths_train)
    
    knn_c = KNeighborsRegressor(n_neighbors=cases_nn)
    knn_d = KNeighborsRegressor(n_neighbors=deaths_nn)
    
    knn_c.fit(X_train, y_cases_train)
    knn_d.fit(X_train, y_deaths_train)
    
    pred_c_rf = regr_c.predict(X_test)
    pred_d_rf = regr_d.predict(X_test)
    
    pred_c_knn = knn_c.predict(X_test)
    pred_d_knn = knn_d.predict(X_test)
    
    pct_c_rf = np.mean(y_cases_test - pred_c_rf) * RDF.POPESTIMATE2018[mask].sum()*100 / RDF[mask].cases.sum()
    pct_d_rf = np.mean(y_deaths_test - pred_d_rf) * RDF.POPESTIMATE2018[mask].sum()*100 / RDF[mask].deaths.sum()
    pct_c_knn = np.mean(y_cases_test - pred_c_knn) * RDF.POPESTIMATE2018[mask].sum()*100 / RDF[mask].cases.sum()
    pct_d_knn = np.mean(y_deaths_test - pred_d_knn) * RDF.POPESTIMATE2018[mask].sum()*100 / RDF[mask].deaths.sum()
    
    apnd_df = pd.DataFrame({'State':st,'Predicted_cases_rf':np.sum(pred_c_rf),'Pct_c_rf':[pct_c_rf],'Predicted_deaths_rf':np.sum(pred_d_rf),'Pct_d_rf':[pct_d_rf],'Predicted_cases_knn':np.sum(pred_c_knn),'Pct_c_knn':[pct_c_knn],'Predicted_deaths_knn':np.sum(pred_d_knn),'Pct_d_knn':[pct_d_knn],'Reported_cases':np.sum(y_cases_test),'Reported_deaths':np.sum(y_deaths_test)})
    state_score = state_score.append(apnd_df)
