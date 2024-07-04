# import pickle
# import json
# import numpy as np
# import os


# __locations = None
# __data_columns = None
# __model = None

# def get_estimated_price(location, sqft, bhk, bath):
#     try:
#         loc_index = __data_columns.index(location.lower())
#     except:
#         loc_index = -1

#     x = np.zeros(len(__data_columns))
#     x[0] = sqft
#     x[1] = bath
#     x[2] = bhk
#     if loc_index >= 0:
#         x[loc_index] = 1

#     return round(__model.predict([x])[0], 2)

# def load_saved_artifacts():
#     print("loading saved artifacts...start")
#     global __data_columns
#     global __locations

#     # Print the current working directory
#     print("Current working directory:", os.getcwd())

#     # Use absolute paths
#     current_dir = os.path.dirname(__file__)
#     columns_path = os.path.join(current_dir, "artifacts/columns.json")
#     model_path = os.path.join(current_dir, "artifacts/banglore_home_prices_model.pickle")

#     # Check if the file exists before attempting to open it
#     if not os.path.exists(columns_path):
#         raise FileNotFoundError(f"File not found: {columns_path}")

#     with open(columns_path, "r") as f:
#         __data_columns = json.load(f)['data_columns']
#         __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

#     global __model
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"File not found: {model_path}")

#     if __model is None:
#         with open(model_path, 'rb') as f:
#             __model = pickle.load(f)
#     print("loading saved artifacts...done")

# def get_location_names():
#     return __locations

# def get_data_columns():
#     return __data_columns

# if __name__ == '__main__':
#     load_saved_artifacts()
#     print(get_location_names())
#     print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
#     print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
#     print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
#     print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location


import pickle
import json
import numpy as np
import os

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    # Print the current working directory
    print("Current working directory:", os.getcwd())

    # Use absolute paths
    current_dir = os.path.dirname(__file__)
    columns_path = os.path.join(current_dir, "artifacts/columns.json")
    model_path = os.path.join(current_dir, "artifacts/updated_banglore_home_prices_model.pickle")  # Updated model path

    # Check if the file exists before attempting to open it
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"File not found: {columns_path}")

    with open(columns_path, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File not found: {model_path}")

    if __model is None:
        with open(model_path, 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
