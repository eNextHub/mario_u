#%%
import pandas as pd


def read_parameter_file(path, sheet_name):

    parameters = {}
    parameter_frame = pd.read_excel(path, sheet_name=sheet_name)

    for parameter, values in parameter_frame.iteritems():

        parameters[parameter] = {
            "Unit": values.iloc[0],
            "Value": values.dropna()[1:].values.tolist(),
        }

    return parameters


# %%
