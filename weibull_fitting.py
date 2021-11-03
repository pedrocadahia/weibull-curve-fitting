# Import required functions
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def read_several_xlsx(filenames, sheet_name_='data'):
    li=[]
    for filename in filenames:
    #     df = pd.read_csv(filename, index_col=None, header=0)
        df = pd.read_excel(filename, sheet_name=sheet_name_)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)

def loss(x, t, y): #function to optimize, basically f(x)-y
    return (x[2]-x[2]*np.exp(-1*(t*x[0])**x[1]))-y

def fit_weibull(x, t): #function we're actually trying to fit
    return x[2]-x[2]*np.exp(-1*(t*x[0])**x[1])

def get_weibull_params(data,vol_column="Volume",grps_column="GRPs",p1=0.05,p2=0.2):       
    x0 = np.array([p1, p2, data[vol_column].max()])
    res_lsq = least_squares(loss, x0, args=(data[grps_column], data[vol_column])) #optimize "loss" function
    return(res_lsq.x)

def get_cuve_type(x):
    output='S-type'
    if x[1]<=1:
        output='C-type'
    return output

def quickadhocfix(x, grps_title='GRPs'):
    x['Volume'] = x['Volume'].astype(float)
    x[grps_title] = x[grps_title].astype(float)
    return(x)

def dict_splits(df, grps_check, print_=False):
    splits = {}
    for iterator in df['iterator'].unique():
        temp = df.loc[(df['iterator']==iterator),]
        temp.dropna(subset=[grps_check],inplace=True)
        temp = temp.loc[:,["Volume",grps_check]]

        if temp.shape[0]>1 and sum(temp.Volume.isna())==0 :
            if print_:
                print('----'+iterator+'----')
                print(temp)
            splits[iterator]=temp
    return(splits)

def grp_cols(df):
    # Logic to create the GRPs column that we in our model
    global grps_check
    if "GRPs" not in df.columns:
        grps_check='GRPs'
        df.insert(df.shape[1],"GRPs",df.filter(like='(Media Unit)').max(axis=1),True)
    else:
        # Some times there is an existent GRPs column, we will use our format which is consistent with this program
        grps_check='GRPs2'
        df.insert(df.shape[1],"GRPs2",df.filter(like='(Media Unit)').max(axis=1),True)
        
def adjust_weibull(df, iterator_columns, grps_check, show_plot=False):
    # Each key is a single "curve" to be adjusted, it should have 4 points, no 0-0 point is needed
    splits = dict_splits(df, grps_check)

    # Saving results
    results = []

    for key in splits:
        try:
            split = splits[key]
            split = split.loc[split.GRPs>0,:]

            # Compute weibull parameters asociate to the data
            weibull_parameters = get_weibull_params(split)

            # Grid to plot the adjusted weibull function
            #     data_domain = np.arange(0,split.GRPs.max()*10)
            data_domain = np.arange(0,split.GRPs.max()*10,(split.GRPs.max())/100)

            # Fit weibull model
            model_weibull_grid = fit_weibull(weibull_parameters, data_domain)

            # Find shape of f(x)
            model_shape = get_cuve_type(weibull_parameters)

            # optimals
            scnd_derivative = np.diff(model_weibull_grid, n=2)/np.max(np.diff(model_weibull_grid, n=2))
            scnd_derivative = np.insert(np.insert(scnd_derivative,0,scnd_derivative[0]),0,scnd_derivative[0])
            optimal_maximum = np.argmin(np.diff(model_weibull_grid, n=2))+2
            try:
                optimal_extreme = np.argmin(scnd_derivative)+np.where(np.round(scnd_derivative[np.argmin(scnd_derivative)::],1)==0)[0][0]
            except:
                optimal_extreme = optimal_maximum
            # Print results Output format expected
            results.append([item for sublist in [key.split('#'), weibull_parameters,[model_shape],[model_weibull_grid[optimal_maximum]],[model_weibull_grid[optimal_extreme]]] for item in sublist])

            # Ploting
            if show_plot:
                print(key, weibull_parameters,model_shape)

                plt.figure()
                plt.scatter(split.GRPs, split.Volume, alpha=0.4)
                plt.plot(data_domain, model_weibull_grid, color='red')
                plt.legend(["Curve Fitting - "+model_shape,"Real Data"],loc="best")
                plt.show()
        except ValueError:
            pass

    results = pd.DataFrame(results, columns=iterator_columns+["a","b","c","shape","optimal","optimal_extreme"])
    return(results)

####################### MAIN ##########################
# PARAMS
filename= 'file_path/filename.xlsx'
iterator_columns = ['Sub-Brand', 'Channel', "Campaign"]
sheet_to_read = "sheet_to_read"

# Read Csv
df = pd.read_excel(filename, sheet_name=sheet_to_read)

# GRP Columns
grp_cols(df)

# Unique Key to iterate with a desired granularity in a single column
df.insert(df.shape[1],"iterator",df[iterator_columns].apply(lambda x: '#'.join(x), axis=1),True)

# Adjust a weibull curve for every split in the dict and get the output
results = adjust_weibull(df, iterator_columns, grps_check, save_plot=True)

# Save Results
results.to_csv(filename.replace(".xlsx","_weibull_curves.csv").replace("Data/","Output/"),sep=';', decimal=',', index = False)
