import pandas as pd
import numpy as np

from collections import Counter

import joblib

import os

np.set_printoptions(suppress=False)

work_dir = r'C:\Users\User\Desktop\python-project-ApartmentPriceAnalysis'
os.chdir(work_dir)

def standardize_missing_values(data):
    
    """
    Replaces custom placeholders for missing values with standard NaN values.

    This function searches the DataFrame for specific strings that are used
    to indicate missing or unavailable information (e.g., 'Zapytaj o cenę',
    'Zapytaj', 'brak informacji') and replaces them with `np.NaN`, which is
    the standard missing value marker in pandas.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame to be cleaned.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with specified placeholder values replaced by NaN.
    """
    missing_placeholders = [
        'Zapytaj o cenę', # 'Ask price'
        'Zapytaj',        # 'Ask'
        'brak informacji' # 'no information'
        ]
    
    return data.replace(missing_placeholders, np.NaN)

def clean_numeric_columns(data):
    
    """
    Cleans and converts specified columns containing numeric values with extra characters to float type.

    This function is designed to process the columns "price", "area", and "rent" in a DataFrame
    where numeric values may be represented as strings containing non-numeric characters such as
    units (e.g., "m²"), letters, or whitespace. The steps include:

    1. Converting each value to string format to enable regex processing.
    2. Removing all alphabetic characters (including Polish-specific letters like 'ł', 'Ł', '²') and spaces.
    3. Replacing commas with dots to correctly format decimal numbers.
    4. Converting cleaned strings to numeric (float) values using `pd.to_numeric`.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame that contains the columns "price", "area", and "rent".

    Returns:
    -------
    pd.DataFrame
        The modified DataFrame with "price", "area", and "rent" columns cleaned and converted to float.
    """
    
    for var in ["price", "area", "rent"]:
        
        data[var] = (
            data[var]
            .astype(str)                              
            .str.replace('[ a-zA-ZłŁ²]*', '', regex=True)
            .str.replace(',', '.', regex=False)
        )
        
        data[var] = pd.to_numeric(data[var])
        
    return data

def categorize_rent(data):
    
    """
    Categorizes rental prices into discrete bins and creates a new column 'rent_cat'.

    This function groups the values from the 'rent' column into five predefined ranges (bins)
    to simplify analysis or modeling. Each bin is assigned a numeric label from 1 to 5.
    The bins are: 
        - 0 to 500
        - 501 to 1000
        - 1001 to 1500
        - 1501 to 2000
        - 2001 and above

    The original 'rent' column is removed after categorization, and the new 'rent_cat'
    column is added with float-typed values.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame that contains a 'rent' column with numeric values.

    Returns:
    -------
    pd.DataFrame
        The modified DataFrame with the 'rent' column replaced by a categorical 'rent_cat' column.
    """
    
    # Group the values from the 'rent' column into ranges
    data['rent_cat'] = pd.cut(data['rent'],
                              bins = [0, 500, 1000, 1500, 2000, np.inf],
                              labels = np.arange(1, 6, 1))
    data['rent_cat'] = data['rent_cat'].astype("float")
    
    # Drop original 'rent' column
    data = data.drop('rent', axis=1)
    
    return data

def process_floor_data(data):
    
    """
    Process floor information in the dataset by extracting and standardizing floor-related features.

    This function performs the following steps:
    1. Extracts the total number of floors in the building from the 'floor' column.
       - Assumes the format is 'apartment_floor/number_of_floors'.
       - If the format does not contain '/', sets the value to NaN.
    2. Extracts the apartment's floor number from the 'floor' column.
       - Converts special floor names ('parter', 'suterena') to '0'.
       - Replaces '> 10' with None (to handle inconsistent data).
    3. If the apartment's floor is labeled as 'poddasze' (attic), replaces it with the total number of floors.
    4. Converts the apartment floor and total floors columns to float type.
    5. Removes the original 'floor' column from the dataframe.

    Parameters:
    ----------
    data : pandas.DataFrame
        The input dataframe containing a 'floor' column with floor information.

    Returns:
    -------
    pandas.DataFrame
        The dataframe with two new columns:
        - 'number_floor_in_building': total floors in the building as float.
        - 'ap_floor': apartment floor number as float.
        The original 'floor' column is dropped.
    """
    
    # Extract apartment floor (left part before '/'), convert special names
    data['number_floor_in_building'] = data['floor'].apply(lambda x: str(x).split('/')[1] if str(x).__contains__('/') else np.NaN).astype('float')
    data['ap_floor'] = data['floor'].apply(lambda x: str(x).split('/')[0]).replace({'parter':'0',
                                                                                    'suterena':'0',
                                                                                    '> 10': None})
    # Replace 'poddasze' (attic) with total floors in building
    data['ap_floor'] = np.where(data['ap_floor'] == 'poddasze',
                                        data['number_floor_in_building'], 
                                        data['ap_floor'])
    # Convert apartment floor to float
    data['ap_floor'] = pd.to_numeric(data['ap_floor'])
    
    # Drop original 'floor' column
    data = data.drop('floor', axis=1)
    
    return data

def fill_missing_categoricals(data):
    
    """
    Fill missing values in selected categorical columns with a placeholder string.

    This function targets a predefined list of categorical columns and replaces any
    missing (NaN) values with the string 'nie podano' (Polish for 'not provided').

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the columns to process.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with missing values in the specified categorical columns
        filled with the placeholder string.
    """
    # single-choice variables in the data
    cols = ['ownership_status', 'flat_condition', 'heating', 'windows', 'mater']
    data[cols] = data[cols].fillna('nie podano') # 'not provided'
    
    return data

def encode_parking_presence(data):
    
    """
    Encode the presence of parking information into a binary indicator column.

    This function creates a new column 'parking_coded' where:
    - 1 indicates that parking information is present (non-missing),
    - 0 indicates that parking information is missing (NaN).

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the 'parking' column.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with an additional 'parking_coded' column representing
        parking presence as a binary indicator.
    """
    
    data['parking_coded'] = data['parking'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    return data

def convert_year_to_int(data):
    
    """
    Convert the 'year' column to integer type, coercing invalid strings to NaN.

    Uses vectorized conversion with pandas.to_numeric, converting
    any non-convertible string to NaN, then casts to nullable integer dtype.

    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame with a 'year' column to be converted.

    Returns:
    -------
    pandas.DataFrame
        DataFrame with 'year' column converted to nullable integer dtype.
    """
    
    data['year'] = pd.to_numeric(data['year']).astype('Int64')
    
    return data

def standardize_ownership_labels(data):
    
    """
    Standardize specific labels in the 'ownership_status' column.

    This function replaces the label 'spółdzielcze własnościowe' with
    the standardized label 'spółdzielcze wł. prawo do lokalu' in the
    'ownership_status' column. All other values remain unchanged.

    Parameters:
    ----------
    data : pandas.DataFrame
        Input DataFrame containing the 'ownership_status' column.

    Returns:
    -------
    pandas.DataFrame
        DataFrame with updated 'ownership_status' labels.
    """
    
    data['ownership_status'] = data['ownership_status'].apply(
        lambda x: 'spółdzielcze wł. prawo do lokalu' if x == 'spółdzielcze własnościowe' else x
    )
    
    return data

def splitcolumn(serieslike, colname, items, missing_categories):
    
    """
    Splits a string column into binary indicator columns for a list of expected items.

    This function processes a string from a specified column in a Series-like object,
    splits it by commas into a list of features, and maps the presence of each expected
    item in that list to a binary value (1 if present, 0 if not). It also tracks any
    unexpected values (not included in `items`) by appending them to the `missing_categories` list.

    Parameters:
    ----------
    serieslike : pandas.Series or pandas.DataFrame
        The data structure containing the column to split.
        
    colname : str
        The name of the column to process.

    items : list of str
        The list of expected categorical values to detect in the column.

    missing_categories : list
        A list that will be extended with unexpected values encountered during processing.

    Returns:
    -------
    pandas.Series
        A Series of binary values (0 or 1) indicating the presence of each item from `items`.
    """
    
    # Extract the value from the given column
    input_value = serieslike[colname]
    
      # If the value is missing, return a Series of 0s (none of the items are present)
    if pd.isna(input_value):
        return pd.Series([0 for x in items])
    
    # Split the string into individual items using commas
    present = input_value.split(',')

    # Remove leading/trailing spaces
    present = [x.strip(' ') for x in present]

    # Filter out empty or very short items (likely noise)
    present = [x for x in present if len(x) > 1]
    
    # Create a binary list indicating whether each expected item is present
    presences = [1 if x in present else 0 for x in items]
    
    # Identify any new/unexpected items not in the `items` list
    new_items = [x for x in present if x not in items]
    if new_items:
        missing_categories.extend(new_items) # Add them to the missing_categories list
    
    return pd.Series(presences)

def multiple_choice_transform(data, train_dataset):
    
    """
    Transforms multiple-choice categorical variables into binary indicator columns.

    This function applies one-hot encoding to columns containing multiple-choice values
    (e.g., "option1, option2") based on predefined expected values stored in a dictionary
    loaded from a joblib file. For each such variable, it creates new binary columns
    indicating the presence of each expected value.

    If unexpected (missing) categories are found in the data and `train_dataset` is True,
    they are collected and printed as a warning.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the multiple-choice categorical variables.

    train_dataset : bool
        Flag indicating whether the function is applied on training data.
        If True, the function will print information about any new, unexpected categories.

    Returns:
    --------
    pandas.DataFrame
        The transformed DataFrame with multiple-choice variables expanded into binary columns.
    """
    
    # Load the predefined dictionary mapping each multiple-choice column
    # to the list of expected values (options)
    var_values_dict = joblib.load("1. Data Preparation/multiple_choice_var_dict.joblib")
    
    # List to collect unexpected (missing) categories encountered in the data
    missing_categories = []
    
    for key in var_values_dict:
        # Prepare names for the new binary columns
        column_names = [key + '_' + x for x in var_values_dict[key]]
        
        # Apply the splitcolumn function row-wise, creating binary indicators
        data[column_names] = data.apply(
            splitcolumn,
            args = (key, var_values_dict[key], missing_categories),
            axis = 1
        )
    
    # Drop the original multiple-choice columns after transformation
    data = data.drop(var_values_dict, axis=1)
    
    # If this is the training dataset, report any unexpected categories found
    if train_dataset:
        if missing_categories:
            category_counts = Counter(missing_categories)
            print("There are new categories that are not in the dictionary:")
            for category, count in category_counts.items():
                print(f"  - {category}: {count} times")
        else:
            print("All the categorizations occurring in the set in multi-vector selection variables were coded.")
    
    return data

def address_transform(addressline, cities_dict):
    
    """
    Extracts region, location (city/town), and street name from a raw address string.

    This function processes a single address line to parse and separate the region,
    city (location), and street components based on commas and a dictionary of cities
    grouped by regions.

    Parameters:
    -----------
    addressline : str or NaN
        A string representing the full address, typically in the format:
        'Street, City, Region'. If missing (NaN), the function returns three NaN values.

    cities_dict : dict
        A dictionary where keys are region names and values are lists of known cities/towns
        in that region. Used to validate which element of the address refers to the location.

    Returns:
    --------
    pandas.Series
        A Series containing three values: [region, location, street].
    """
    
    # Handle missing or null addresses
    if pd.isna(addressline):
        return pd.Series([np.nan, np.nan, np.nan])
    
    # Split the address by commas and strip whitespace
    parts = [x.strip() for x in addressline.split(',')]
    
    # Extract the region (last element)
    region = parts[-1]
    
    # Attempt to identify location (city/town) using cities_dict
    city_in_region = cities_dict[region]
    
    if parts[-2] in city_in_region:
        location = parts[-2]
    elif len(parts) >= 3 and parts[-3] in city_in_region:
        location = parts[-3]
    else:
        location = np.nan

    # Determine if the first part of the address is a valid street
    potential_street = parts[0]
    if potential_street == location:
        street = np.nan
    else:
        # Remove common prefixes from street names
        street = potential_street
        for prefix in ['ul. ', 'al. ', 'pl. ']:
            street = street.removeprefix(prefix)
        street = street.strip()

    return pd.Series([region, location, street])

def location_transform(data):
    
    """
    Extracts structured geographic information (region, location, and street/district)
    from a raw address column using a reference dataset of Polish localities.

    This function reads a CSV file containing Polish place names, filters valid settlement types,
    and constructs a dictionary that maps regions (voivodeships) to cities and villages.
    It then uses this dictionary to parse the 'address' column and extract three elements:
    region, location (city/village), and street/district, which are added as new columns.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing a column named 'address' with full address strings.

    Returns:
    --------
    pandas.DataFrame
        The same DataFrame with three new columns:
        - 'region': the voivodeship (region) the address belongs to,
        - 'location': the specific city/town/village found in the address,
        - 'street/district': the remaining part of the address (typically street).
    """

    # Load external CSV file containing Polish place names and administrative regions
    locations = pd.read_csv('1. Data Preparation/locations_and_regions.csv')

    # Keep only rows with relevant types of settlements
    valid_types = ['wieś', 'miasto', 'osada', 'kolonia', 'osada leśna']
    locations = locations[locations['Rodzaj'].isin(valid_types)]

    # Group places by region (voivodeship)
    locations = locations.groupby('Województwo', axis=0)

    # Create a dictionary: region -> list of cities/villages in that region
    cities_dict = {}
    for reg in locations:
        cities_dict[reg[0]] = list(reg[1]['Nazwa miejscowości'])

    # Manually correct known naming mismatch: Stargard (used to be Stargard Szczeciński)
    cities_dict['zachodniopomorskie'].append('Stargard')

    # Apply address transformation to extract region, location, and street/district
    data[['region', 'location', 'street/district']] = data['address'].apply(
        address_transform,
        args=[cities_dict]
    )

    return data

def city_info_transform (data):
        
    """
    Enriches the dataset with demographic and administrative information about cities.

    This function merges the input dataset with external city-level statistics, such as
    population size, population density, and administrative rights (powiat rights).
    It also creates categorized bins for population size and density to facilitate analysis.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame that must include 'location' and 'region' columns,
        typically produced by the `location_transform` function.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with additional features:
        - 'pop_numb_cat': population size category (0-7), where:
              0 → up to 10,000 people  
              1 → 10,001-20,000  
              2 → 20,001-50,000  
              3 → 50,001-100,000  
              4 → 100,001-250,000  
              5 → 250,001-500,000  
              6 → 500,001-1,000,000  
              7 → more than 1,000,000
        - 'pop_dens_cat': population density category (0-7), where:
              0 → up to 500 people/km²  
              1 → 501-1000  
              2 → 1001-1500  
              3 → 1501-2000  
              4 → 2001-2500  
              5 → 2501-3000  
              6 → 3001-3500  
              7 → more than 3500
        - 'with_powiat_rights': binary indicator (0 or 1) whether the city has powiat (county-level) administrative rights.

        The function also drops unneeded columns from the merged location data.

    """    
    
    # Load city-level demographic and administrative data    
    locations_data = pd.read_excel("1. Data Preparation/locations_info.xlsx")
    
    # Categorize population size into 8 bins
    locations_data['pop_numb_cat'] = pd.cut(
        locations_data['Liczba ludności'],
        bins=[0, 10_000, 20_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000],
        labels=np.arange(0, 8)
    )
    
    # Categorize population density into 8 bins (step = 500)
    locations_data['pop_dens_cat'] = pd.cut(
        locations_data['Gęstość zaludnienia'],
        bins=range(1, 4001, 500),
        labels=np.arange(0, 8)
    )
    
    # Merge with the main dataset using location and region
    data_merged = data.loc[:, 'link':'street/district'].merge(
        locations_data,
        left_on=['location', 'region'],
        right_on=['Miasto', 'Województwo'],
        how='left'
    )
    
    # Replace NaNs in powiat rights with 0 (default: no rights)    
    data_merged['with_powiat_rights'] = data_merged['na_prawach_powiatu'].fillna(0)
    
    # Convert categories to numeric and fill missing values with 0
    # Missing data refers to villages, i.e., the file contains data only for cities
    data_merged['pop_numb_cat'] = pd.to_numeric(data_merged['pop_numb_cat']).fillna(0)
    data_merged['pop_dens_cat'] = pd.to_numeric(data_merged['pop_dens_cat']).fillna(0)

    # Remove unnecessary columns from location metadata
    data_merged.drop([
        'Miasto', 'Powiat', 'Województwo', 'Powierzchnia',
        'Liczba ludności', 'Gęstość zaludnienia', 'na_prawach_powiatu'
    ], axis=1, inplace=True)
    
    return data_merged

def preliminary_transform (data, train_dataset):
    
    """
    Performs a full preliminary transformation pipeline on the input dataset.

    This function executes a sequence of data cleaning and feature engineering steps
    that prepare the dataset for further analysis or modeling. It ensures consistency
    in missing values, encodes categorical data, standardizes formats, and enriches
    the data with external geographic and demographic information.

    Parameters:
    -----------
    data : pandas.DataFrame
        The raw dataset to be processed. Must contain all necessary columns 
        such as 'floor', 'year', 'address', etc.

    train_dataset : bool
        Indicates whether the data being processed is training data.
        This is used to display warnings about unexpected new categories
        in multi-label (multi-choice) variables.

    Returns:
    --------
    pandas.DataFrame
        A cleaned and enriched DataFrame ready for modeling or further processing.

    The pipeline performs the following transformations:
    ----------------------------------------------------
    1. `standardize_missing_values` - Converts placeholders like "Zapytaj o cenę" to NaN.
    2. `clean_numeric_columns` - Cleans and converts numeric columns such as 'price', 'area', 'rent'.
    3. `categorize_rent` - Categorizes rental prices into bins.
    4. `process_floor_data` - Splits and normalizes apartment floor info.
    5. `fill_missing_categoricals` - Fills missing categorical values with "nie podano".
    6. `encode_parking_presence` - Encodes binary presence of parking (e.g., yes/no).
    7. `convert_year_to_int` - Converts the 'year' column is of integer type.
    8. `standardize_ownership_labels` - Standardizes ownership labels (e.g., unifying similar terms).
    9. `multiple_choice_transform` - One-hot encodes multi-choice variables based on a predefined dictionary.
    10. `location_transform` - Extracts region, city, and street from the address.
    11. `city_info_transform` - Adds population and administrative info by merging with external city data.
    """
    
    standardized_missing_values_data = standardize_missing_values(data)
    cleaned_numeric_columns_data = clean_numeric_columns(standardized_missing_values_data)
    categorized_rent_data = categorize_rent(cleaned_numeric_columns_data)
    processed_floor_data = process_floor_data(categorized_rent_data)
    fill_missing_categoricals_data = fill_missing_categoricals(processed_floor_data)
    encoded_parking_presence_data = encode_parking_presence(fill_missing_categoricals_data)
    converted_year_to_int_data = convert_year_to_int(encoded_parking_presence_data)
    standardized_ownership_labels_data = standardize_ownership_labels(converted_year_to_int_data)
    multiple_choice_transformed_data = multiple_choice_transform(standardized_ownership_labels_data,
                                                                 train_dataset)
    location_transformed_data = location_transform(multiple_choice_transformed_data)
    city_info_transformed_data = city_info_transform(location_transformed_data)
    
    return city_info_transformed_data