import os
import pandas as pd

__dir__ = os.path.abspath(os.path.dirname(__file__))

def load_fennell2023():
    '''
    Load data from experiment one in Fennell and Ratcliff (2023).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data with columns: 'subjectNumber', 'blockNumber', 'trialNumber', 'rt', 'numberOfStimulus', 'responseError'
    '''

    data_path = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "data", "Fennell2023_exp1.csv")

    data = pd.read_csv(data_path, index_col=0)
    return data
