"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np

class Patient:

    def __init__(self, name, weight, height):
        self.name = name
        self.weight = weight
        self.height = height
    
    def get_body_mass_index(self):
        """Compute body mass index: weight_in_kg / height_in_meters**2
        """
        return self.weight / self.height**2

def load_csv(filename):  
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data: np.array) -> np.array:
    """Calculate the daily mean of a 2d inflammation data array.

    :param data: 
        expects a table (2d array) where each row contains measurements for a single 
        patient taken over a number of days
    :return:
        an array with average values
    """
    return np.mean(data, axis=0)


def daily_max(data: np.array) -> np.array:
    """Calculate the daily max of a 2d inflammation data array.

    :param data:
        expects a table (2d array) where each row contains measurements for a single 
        patient taken over a number of days
    :return:
        an array with maximum values
    """
    return np.max(data, axis=0)


def daily_min(data: np.array) -> np.array:
    """Calculate the daily min of a 2d inflammation data array.

    :param data:
        expects a table (2d array) where each row contains measurements for a single 
        patient taken over a number of days
    :return: 
        an array with minimum values
    """
    return np.min(data, axis=0)

