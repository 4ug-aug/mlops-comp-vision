import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import *

def make_report(reference_data=None, current_data=None):

    if reference_data is None:
        print("No reference data provided")
    if current_data is None:
        print("No current data provided")

    # Make column names of current data match reference data
    current_data = current_data.drop(columns=['time'])
    current_data.columns = reference_data.columns


    data_test = TestSuite(tests=[TestNumberOfMissingValues(), TestNumberOfMissingValues()])
    data_test.run(reference_data=reference_data, current_data=current_data)
    return Report(data_test)
