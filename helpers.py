import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from numpy import random


##############
#### Data preparation

def merge_specific_data(all_data_dict, column_name):
    """
    Merge a specific column of data from each of the three cities into a single dataframe

    Args:
        all_data_dict (dict): dictionary of dataframes containing data for each city
        column_name (str): Name of column in all_data_dict corresponding to required data 

    Returns:
        specific_merged: Merged dataframe
    """

    if isinstance(all_data_dict, dict)==False:
        raise ValueError(f'{all_data_dict} must be a dictionary of the data')

    # Get city names from the dictionary of the city data
    cities = list(all_data_dict.keys())
    
    # Checks if column_name is in dataframe
    if column_name not in list(all_data_dict[cities[0]].keys()):
        raise ValueError(f' \'{column_name}\' not a column name in all_data_dict')
   
    # Copy needed so that original dataframe remains unchanged?
    specific_merged = all_data_dict[cities[0]][column_name].copy()
    
    # Merge into one dataframe
    num_cities = len(cities)
    for i in range(1, num_cities):
        specific_merged = pd.merge(specific_merged, all_data_dict[cities[i]][f'{column_name}'], left_index=True, right_index=True)
        
    # set column titles to the city names
    specific_merged.columns = cities
    
    return specific_merged
    

def get_unique_conds(df):
    """
    Return all unique coniditions from a provided dataframe

    Args:
        df (df): Dataframe to get unique conditions from

    Returns:
        unsorted_conditions (array): Unsorted conditions
    """
    
    try:
        df = pd.DataFrame(df)
    except:
        raise ValueError(f'{df} must be a dataframe or convertable to a dataframe')
    
    # Stack into one df
    stacked_merge = df.stack()
    
    # get unique conditions
    unsorted_conditions = stacked_merge.unique()
    
    return unsorted_conditions


def assign_colours(conditions, colours_list=None):
    """
    Assign colours to each unique condition

    Args:
        conditions (list): Conditions which will each be assigned a colour
        colours_list (list, optional): list of colours to assign. Defaults to None.

    Returns:
        colours (dict): Dictionary of each condition and their colour
    """
      
    # Create list if no list provided 
    if colours_list is None:
        colours_list = []

    # Check conditions is (or can be converted to) a list
    try:
        conditions = list(conditions)
    except:
        raise ValueError(f'{conditions} must be list-like ')
    
    # Check colours_list is (or can be converted to) a list
    try:
        colours_list = list(colours_list)
    except:
        raise ValueError(f'{colours_list} must be list-like ')

    # If more colours needed, append default colours onto end of colours list
    if len(conditions) > len(colours_list):
        diff = len(conditions) - len(colours_list)
        for i in range(0, diff):
            colours_list.append(f'C{i}')

    # Assign colours to each of the unique weather conditions for later plotting
    colours = {}
    for i, condition in enumerate(conditions):
        
        # Append the colour if it is valid, else return a valueerror 
        if is_color_like(colours_list[i]):
            colours[condition] = colours_list[i] 
        else:
            raise ValueError(f'{colours_list[i]} is not a valid colour')
        
    return colours



##################
#### Calculating probability arrays

def calc_discrete_probability_array(all_specific, possible_conditions):
    """
    Calculate probability array for some given (discrete) data 

    Args:
        all_specific (pd.Series): All data for a given type of data (eg summary)
        possible_conditions (list): The possible conditions for this data, given in order to be displayed

    Returns:
        probability_array: array of probabilities of each condition occuring after each previous condition
        probability_df: probability_array as a dataframe
    """

    # Check data is in list format or can be converted to a list.
    try:
        all_conditions = list(all_specific)
    except:
        raise ValueError(f'{all_specific} must be list-like')

    # Check possible_conditions is (or can be converted to) a list
    try:
        possible_conditions = list(possible_conditions)
    except:
        raise ValueError(f'{possible_conditions} must be list-like ')

    # Number of past conditions to look at
    # This cannot be changed in this 
    past_hours=1
 
    n_unique_conditions = len(possible_conditions)
    
    all_conditions = list(all_specific)
    # all_conditions = (all_specific).tolist()  
    num_conditions = len(all_conditions) 

    # Create an array of number of times a certain weather condition follows another weather condition
    array_counter = np.zeros(shape=(n_unique_conditions, n_unique_conditions), dtype=float)

    # fill array
    for start_index in range(0, num_conditions - past_hours):
        
        # Find current weather condition 
        current_weather_conditions = all_conditions[start_index]
        # Find index of current weather condition
        current_condition_index = possible_conditions.index(current_weather_conditions)
        
        # Get index for next weather condition
        next_index = start_index + past_hours
        # Determine next weather condition
        next_weather_condition = all_conditions[next_index]
        # Find index of this condition in the array
        next_condition_index = possible_conditions.index(next_weather_condition)
        
        # Increase counter for this match. 
        array_counter[current_condition_index, next_condition_index] += 1.0

    # Normalise array - make into probabilites
    probability_array = normalise_array(array_counter)
    
    # Produce results as dataframe
    probability_df = pd.DataFrame(probability_array, columns=[possible_conditions], index=possible_conditions)
    probability_df
    
    return probability_array, probability_df


def plot_probabilities(city_prob_array, ax, unique_conds, title=None, colourbar=False):
    """
    Plot a probability array as a colourmap

    Args:
        city_prob_array (array): array of probabilites to be plotted
        ax: axis for colourmap to be plotted onto
        unique_conds (list): all possible conditions for this city
        title (str, optional): Optional title for figure. Defaults to None.
        colourbar (bool, optional): Option to plot colourbar or not. Defaults to False.
        
    """
    
    # Check probability array is (or can be converted to) an array
    try:
        city_prob_array = np.array(city_prob_array)
    except:
        raise ValueError(f'{city_prob_array} must be an array or able to convert to an array ')
    
    
    # Check conditions is (or can be converted to) a list
    try:
        unique_conds = list(unique_conds)
    except:
        raise ValueError(f'{unique_conds} must be list-like ')   
       
       
    # The choice of colour allows clear distinction between high probabilities and low probabilities.
    im = ax.imshow(city_prob_array, aspect="auto", cmap='binary', vmax=1, vmin=0)

    ax.set_xlabel("Next condition")
    ax.set_ylabel(f"Past condition")
    
    if title is not None:
        ax.set_title(f"{title}")
    
    # Label the unique conditions
    ax.set_xticks(np.arange(0,len(unique_conds), 1))
    ax.set_xticklabels(unique_conds)
    ax.set_yticks(np.arange(0,len(unique_conds), 1))
    ax.set_yticklabels(unique_conds)

    # Makes both axes increase 
    ax.invert_yaxis()
    
    if colourbar==True:
        colourbar = plt.colorbar(im, orientation='horizontal', location='bottom', pad=0.1)  
        colourbar.ax.set_xlabel("Probabilities")


def normalise_array(array_count):
    """
    Normalise each row in an array so that the values in each row sum to 1.
    
    Parameters:
        array_count (array): array of counts 
    Returns:
        array: array with normalised rows, where each row sums to 1.
    """
    
    # Check array_count is (or can be converted to) an array
    try:
        array_count = np.array(array_count)
    except:
        raise ValueError(f'{array_count} must be an array or able to convert to an array ')
    
    
    # Calculate the sum of each row
    row_sums = np.sum(array_count, axis=1).reshape(-1, 1)
    
    # Avoid division by zero by setting any zero row sums to 1
    row_sums[row_sums == 0] = 1
    
    # Normalise each row by dividing by the row sum
    normalised_matrix = array_count / row_sums
    return normalised_matrix


##########
### Predicting and plotting data

def data_prediction(prob_array, num_predictions, possible_conditions, initial_selection=None):
    """
    Produces weather prediction for a city
    
    Args:
        prob_array (array): array of probabilites to be used in prediction
        num_predictions (int): length of weather prediction model 
        possible_conditions (list): all possible weather conditions for this city
        initial_selection (str, optional): initial weather prediction. Defaults to None - assigned randomly.

    Returns:
        data_predictions: list of weather predictions for this city.
    """
    
    # Ensure num_predictions is a valid input
    try:
        num_predictions = int(num_predictions)
    except:
        raise ValueError('Number of predictions must be an integer')
  
    if num_predictions <= 0:
        raise ValueError('Number of predictions must be a positive integer')
    
    if num_predictions > 1000:
        raise ValueError('Input fewer than 1000 samples')

    # Check possible_conditions is (or can be converted to) a list
    try:
        possible_conditions = list(possible_conditions)
    except:
        raise ValueError(f'{possible_conditions} must be list-like ')

    # Check prob_array is (or can be converted to) an array
    try:
        prob_array = np.array(prob_array)
    except:
        raise ValueError(f'{prob_array} must be an array ')

    # Probability array
    probabilities = prob_array

    rng = random.default_rng()

    # Represent each condition as integers, eg from clear=0 to snow=max - easier indexing
    int_values = np.arange(0, len(possible_conditions), 1, dtype=int)

    # Empty array of predictions to be filled 
    prediction = np.zeros(shape=num_predictions, dtype=int)

    # Use initial selection if provided and valid.
    if initial_selection is not None and initial_selection in possible_conditions:
        initial_selection = possible_conditions.index(initial_selection)
        
    # Else use a random initial value from values that have occured in a dataset 
    else:
        # If condition did not occur in data set, its valid_input_probability will be 0 so won't be picked as a starting condition.
        valid_input_probabilities = np.sum( prob_array, axis=1 )
        initial_selection = rng.choice(int_values, p=valid_input_probabilities) 
    
    prediction[0] = initial_selection

    # Select the current state and use associated probabilities to choose the new state
    for i in range(1, num_predictions):
        current_weather = prediction[i-1]
        probability = probabilities[current_weather]
        
        # Randomly pick next weather condition using provided probabilities as weightings
        new_weather = rng.choice(int_values, p=probability)
        # Input new value
        prediction[i] = new_weather

    # Replace integers with the weather condition names again
    data_predictions = []
    for condition in prediction:
        data_predictions.append(possible_conditions[condition]) 

    return data_predictions



def multiple_data_predictions(prob_array, samples, num_predictions, possible_conditions, initial_selection=None):
    """
    Produce multiple weather predictions for a city

    Args:
        prob_array (array): array of probabilites to be used in prediction
        samples (int): Number of runs to make
        num_predictions (int): Length of a singular weather prediction 
        possible_conditions (list): all distinct conditions for this city

    Returns:
        predictions_list: list of all the weather predictions for each sample.
    """
    
    # If samples is not an int (and can't be converted to an int), raise value error
    try:
        samples = int(samples)
    except:
        raise ValueError('Samples must be an integer')
  
    if samples <= 0:
        raise ValueError('Samples must be a positive integer')
    
    if samples > 100:
        raise ValueError('Input fewer than 100 samples')
        
    predictions_list = []
    for i in range(0, samples):
        results =  data_prediction(prob_array, num_predictions, possible_conditions=possible_conditions, initial_selection=initial_selection)
        predictions_list.append(results)
        
    return predictions_list


def plot_predictions(predictions_list, colours, axis, plot_legend=False, title=None, y_label=None):
    """
    Plot multiple weather predictions for a city

    Args:
        predictions_list (list): Multiple predictions of the weather for each city 
        colours (dict): Colours corresponding to each unique weather condition
        axis : Axis for runs to be plotted on
        plot_legend (bool, optional): If true, legend is plotted below this axis. Defaults to False.
    """
    
    # Check predictions_list is (or can be converted to) a list
    try:
        predictions_list = list(predictions_list)
    except:
        raise ValueError(f'{predictions_list} must be list-like ')
    
    if isinstance(colours, dict)==False:
        raise ValueError(f'{colours} must be a dictionary')
    
    ax=axis
    
    # The unique conditions are contained within the colour dictionary
    possible_conditions = list(colours.keys())

    # Name all the runs
    runs = []
    for i in np.arange(0, len(predictions_list), 1, dtype=int):
        runs.append(f'Run {i+1}')
    # runs = [f"Run {i+1}"  for i in np.arange(0, len(predictions_list), 1, dtype=int)]

    nruns = len(runs)
    len_predictions= len(predictions_list[0])

    shift = 1
    
    hours_forecast = np.arange(1, len_predictions + 1, 1, dtype=int)
    
    # If only a single sample, show the line graph of the prediction for clearer transitions
    if nruns == 1:
        
        weather_ints = np.arange(0, len(possible_conditions), dtype=int)

        # Fill in integer values corresponding to the weather conditions
        int_values = np.zeros(len_predictions)
        for i, prediction in enumerate(predictions_list[0]):
            int_values[i] = possible_conditions.index(prediction) 

        # Plot the predicted result
        ax.plot(hours_forecast, int_values)
        
        # Used to shift colour bar below the line graph
        shift = -0.2
            
        # Set y ticks to show each weather condition
        ax.set_yticks(weather_ints)
        ax.set_yticklabels(possible_conditions)
        ax.set_ylim(-0.5, weather_ints[-1] + 0.2)

        ax.grid()

    # PLot samples as colour bars
    for i, predictions in enumerate(predictions_list):
        
        # Used to plot runs regularly on top of each other
        if nruns != 1:
            shift = i
            
        # Set colour of the marker based on the state for each predicted value
        coloured_predictions = [] 
        for prediction in predictions:
            coloured_predictions.append(colours[prediction] )

        run = np.ones(shape=len_predictions) * shift
        ax.scatter(hours_forecast, run, s=10, marker="s", color=coloured_predictions)

    ax.set_xlabel('Prediction number')
    
    if y_label is not None:
        ax.set_ylabel(f'{y_label}')

    if nruns != 1:
        # Set y ticks to label each of the runs
        ax.set_yticks(np.arange(0, nruns, 1))
        ax.set_yticklabels(runs)
    
    if title is not None:
        ax.set_title(f'{title}')
    
    # Adds legend underneath figure 
    if plot_legend:
        # Shows colour of each square
        legend_handles = [mpatches.Patch(color=colour, label=label) for label, colour in colours.items()]
        ax.legend(handles=legend_handles, ncol=len(colours), loc='upper center', bbox_to_anchor=(0.5, -0.15));
    

###############
### Following functions are used to calculate the variability of the weather in each city

def compare_variations(cities, prob_arrays, num_runs, num_predictions, possible_conditions, initial_selection):
    """
    Calculate variability of weather in cities

    Args:
        cities (_type_): list of cities
        prob_arrays (dict): dictionary of probabiliy arrays for each cities
        num_runs (_type_): Number of times to run
        num_predictions (_type_): Number of predictions in a single run
        possible_conditions (list): all distinct conditions for this city
        initial_selection (str): starting condition for predictions. 

    Returns:
        _type_: _description_
    """
    
    # Check cities is (or can be converted to) a list
    try:
        cities = list(cities)
    except:
        raise ValueError(f'{cities} must be list-like ')
    
    predictions_list = []
    av_variation_dict = {}
    
    for city in (cities):

        predictions_list = multiple_data_predictions(prob_array=prob_arrays[f'{city}'], samples=num_runs, 
                                                     num_predictions=num_predictions, possible_conditions=possible_conditions, initial_selection=initial_selection)
        
        num_variations = av_variations(predictions_list)    
        
        av_variation_dict[f'{city}'] = num_variations
        
    return av_variation_dict


def av_variations(data_list):
    """
    Calculate average number of variations in a list of lists

    Args:
        data_list (list): list of lists of data.

    Returns:
        average (float): Average number of switches in conditions
    """    
    
    # Check data_list is (or can be converted to) a list
    try:
        data_list = list(data_list)
    except:
        raise ValueError(f'{data_list} must be list-like ')
    
    sum = 0
    for i in range(len(data_list)):
        counter = variation_counter(data_list[i])
        sum += counter
    average = sum / (i+1)
    return average


def variation_counter(list):
    """
    Counts number of times the conditions switch in a list

    Args:
        list (_type_): list of data

    Returns:
        counter (int): number of times conditions switched. 
    """
    counter = 0
    for i in range(1, len(list)):
        if list[i] != list[i-1]:
            counter += 1
    return counter    



###############
### Following functions are used for continuous datasets

def calc_cont_prob_arrays(city_data, cities_data):
    """
    Calculate probability array for continuous data

    Args:
        city_data (list_like): data for which probability array will be calculated for. 
        cities_data (array): Unique data across all the cities. Used to determine bins

    Returns:
        probability_array: array of probabilites 
        probability_df: dataframe of probabilites with labels
        bin_indices: array of the indices of the bins that each datapoint falls into
        bins: upper bounds of bins.  
    """

    # Check city_data is in list format or can be converted to a list.
    try:
        city_data = list(city_data)
    except:
        raise ValueError(f'{city_data} cannot be converted to a list')

    # Check data is in list format or can be converted to a list.
    try:
        cities_data = list(cities_data)
    except:
        raise ValueError(f'{cities_data} cannot be converted to a list')

    # Number of past conditions to look at
    past_hours=1
    
    # Split into bins and get the upper values of the bins and put each bit of data into the bin it belongs to. 
    bin_indices, bin_names = get_bins(city_data, cities_data=cities_data)
        
    num_bins = len(bin_names)

    # Empty array to be filled
    array_counter = np.zeros(shape=(num_bins,num_bins), dtype=int)

    for i in range(0, len(city_data)-past_hours):
        start_index = i
        next_index = i+past_hours
        
        # Find bin of current condition
        current_condition_index = bin_indices[start_index]
        
        # Find bin of next condition
        next_condition_index = bin_indices[next_index]

        # Increase counter for this match. 
        array_counter[current_condition_index, next_condition_index] += 1.0

    # Normalise array - make into probabilites
    probability_array = normalise_array(array_counter)
    
    # Produce results as dataframe
    probability_df = pd.DataFrame(probability_array, columns=bin_names, index=bin_names)
    probability_df
    
    return probability_array, probability_df, bin_indices, bin_names


def get_bins(data_to_bin, cities_data, num_bins = 10):
    """
    Get values for bins

    Args:
        data_to_bin (list): List of data being split into bins 
        cities_data (list): Unique data across all the cities. Used to determine bins
        num_bins (int, optional): Number of bins. Defaults to 10.

    Returns
        bin_indices: array of the indices of the bins that each datapoint falls into
        bins: upper bounds of bins.  
    """

    ### rewrite
    # Use this to split bins up so that they have the same number of temperature points in each across the cities
    temp_df = pd.qcut(cities_data, q=num_bins, duplicates='drop')

    # Now get the counts for each bin to verify
    bin_counts = pd.DataFrame(temp_df.value_counts().sort_index())
    
    # Creates list/array of the upper bounds of each bin
    # 0.001 added to ensure the maximum data point falls within the defined bins. 

    bins = []
    for bin_upper_bound in bin_counts.index:
        bins.append(bin_upper_bound.right+0.001)
    bins = list( np.round(bins, 3) )
    bins 
    
    bin_indices = np.digitize(data_to_bin, bins )
    
    return bin_indices, bins



