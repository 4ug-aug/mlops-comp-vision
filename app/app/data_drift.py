import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import *

def make_report(reference_data=None, current_data=None):

    if reference_data is None:
        print("No reference data provided")
        # Look for reference database in app/reference_database.csv
        try:
            reference_data = pd.read_csv('app/reference_database.csv')
        except:
            print("No reference data found")
        return None
    if current_data is None:
        print("No current data provided")
        # Look for current database in app/prediction_database.csv
        try:
            current_data = pd.read_csv('app/prediction_database.csv')
        except:
            print("No current data found")
            return None

    # Load reference data
    reference_data = pd.read_csv(reference_data)
    # Load current data
    current_data = pd.read_csv(current_data)

    # Make column names of current data match reference data
    current_data = current_data.drop(columns=['timestamp'])
    reference_data = reference_data.drop(columns=['timestamp'])


    data_test = TestSuite(tests=[TestNumberOfMissingValues(), TestNumberOfMissingValues()])
    data_test.run(reference_data=reference_data, current_data=current_data)
    return Report(data_test)
