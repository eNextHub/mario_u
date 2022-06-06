#%%
from functools import cached_property
import pandas as pd
import numpy as np
from mario_u.tools.constansts import (
    DEFAULT,
    SHEETS_TO_READ,
    SHEETS,
    HEADERS,
    TECH_TYPE,
    INDEX,
)
from copy import deepcopy


def read_parameter_file(path, sheet_name):

    parameters = {}
    parameter_frame = pd.read_excel(path, sheet_name=sheet_name)

    for parameter, values in parameter_frame.iteritems():

        parameters[parameter] = {
            "Unit": values.iloc[0],
            "Value": values.dropna()[1:].values.tolist(),
        }

    return parameters


class SetReader:
    def __init__(self, file):

        if isinstance(file, str):
            excel = pd.ExcelFile(file)

            self.file = {
                sheet: excel.parse(
                    sheet_name=SHEETS[sheet].sheet_name,
                    index_col=SHEETS[sheet].index_col,
                    header=SHEETS[sheet].header,
                )
                for sheet in SHEETS_TO_READ.main_sets
            }

        elif isinstance(file, dict):
            self.file = file

        else:
            raise ValueError("wrong format")

    def technology_need(self):

        data = self.file["technology_need"]

        # backup for next step
        self._tn = deepcopy(data)

        new_cols_id = [
            "activity_name",
            "activity_unit",
            "commodity_name",
            "commodity_unit",
        ]
        new_data = pd.DataFrame(
            index=data.index,
            columns=pd.MultiIndex.from_tuples([HEADERS[i] for i in new_cols_id]),
        )

        for row, cols in data.iterrows():

            technology_name = HEADERS.technology_name
            technology_unit = HEADERS.technology_unit
            need_name = HEADERS.need_name
            need_unit = HEADERS.need_unit

            renamed_info = self._rename_technology_need(
                technology_name=cols[technology_name],
                need_name=cols[need_name],
                need_unit=cols[need_unit],
            )

            for col in new_cols_id:
                new_data.loc[row, HEADERS[col]] = renamed_info[col]

        return pd.concat([data, new_data], axis=1)

    def factors(self):

        return self.file["factors"]

    def satellite(self):

        return self.file["satellite"]

    def information(self):

        return self.file["information"]

    def investment_time(self):

        data = self.file["investment_time"]

        return pd.date_range(
            start=data.loc[HEADERS.start_date].values[0],
            end=data.loc[HEADERS.end_date].values[0],
            freq=data.loc[HEADERS.frequency].values[0],
        )

    def operation_time(self):

        data = self.file["operation_time"]

        return pd.date_range(
            start=data.loc[HEADERS.start_date].values[0],
            end=data.loc[HEADERS.end_date].values[0],
            freq=data.loc[HEADERS.frequency].values[0],
        )

    def _rename_technology_need(self, technology_name, need_name, need_unit):

        return {
            "activity_name": "Exploiting {} for {}".format(technology_name, need_name),
            "activity_unit": need_unit,
            "commodity_name": "{} from {}".format(need_name, technology_name),
            "commodity_unit": need_unit,
        }

    def solar_activities(self):
        filtered_data = self._tn.loc[
            self._tn[HEADERS.technology_type] == TECH_TYPE.solar, :
        ]
        new_cols_id = [
            "activity_name",
            "activity_unit",
            "commodity_name",
            "commodity_unit",
            "technology_name",
            "technology_unit",

        ]
        new_data = pd.DataFrame(
            index=filtered_data.index,
            columns=pd.MultiIndex.from_tuples([HEADERS[i] for i in new_cols_id]),
        )

        technology_name = HEADERS.technology_name

        for row, cols in filtered_data.iterrows():
            renamed_info = {
                "activity_name": "Selling {} surpluss".format(cols[technology_name]),
                "activity_unit": TECH_TYPE.solar,
                "commodity_name": "Sold surpluss from {}".format(cols[technology_name]),
                "commodity_unit": TECH_TYPE.solar,
                "technology_name": cols[technology_name],
                "technology_unit": cols[HEADERS.technology_unit]
            }
            for col in new_cols_id:
                new_data.loc[row, HEADERS[col]] = renamed_info[col]

        return new_data.drop_duplicates()
    def storage_activities(self):

        filtered_data = self._tn.loc[
            self._tn[HEADERS.technology_type] == TECH_TYPE.storage, :
        ]
        new_cols_id = [
            "activity_name",
            "activity_unit",
            "commodity_name",
            "commodity_unit",
            "satellite_name",
            "satellite_unit",
            "technology_name",
            "technology_unit",

        ]
        new_data = pd.DataFrame(
            index=filtered_data.index,
            columns=pd.MultiIndex.from_tuples([HEADERS[i] for i in new_cols_id]),
        )

        technology_name = HEADERS.technology_name

        for row, cols in filtered_data.iterrows():
            renamed_info = {
                "activity_name": "Charging {}".format(cols[technology_name]),
                "activity_unit": TECH_TYPE.storage,
                "commodity_name": "{} Charge".format(cols[technology_name]),
                "commodity_unit": TECH_TYPE.storage,
                "satellite_name": "Charge from {} ".format(cols[technology_name]),
                "satellite_unit": TECH_TYPE.storage,
                "technology_name": cols[technology_name],
                "technology_unit": cols[HEADERS.technology_unit]
            }
            for col in new_cols_id:
                new_data.loc[row, HEADERS[col]] = renamed_info[col]

        return new_data.drop_duplicates()

    def add_new_sets(
        self,
        df:pd.DataFrame
        ):

        items = ['need_name','need_unit',"technology_name","technology_unit","technology_type"]

        columns = pd.Index([HEADERS[item] for item in items])

        if not df.columns.equals(columns):
            raise ValueError('wrong columns')



        new_sets = SetReader({"technology_need":df})

        return {
            "technology_need" : new_sets.technology_need(),
            "storage_activities": new_sets.storage_activities(),
            "solar_activities"  : new_sets.solar_activities(),
        }

    def add_storages_to_satellite(self,original_satellite,storages):
        return original_satellite.append(storages[[HEADERS.satellite_name,HEADERS.satellite_unit]])


class InputBuilder:
    def __init__(self, data_factory):
        self.data_factory = data_factory

    @property
    def G(self):
        G = pd.DataFrame(
            data=0,
            index=pd.Index(self.data_factory.needs, name=INDEX.need),
            columns=pd.Index(self.data_factory.commodities, name=INDEX.commodity),
        )

        for need in self.data_factory.needs:
            correspondant_commodities = self.data_factory.find_commodity_by_need(need)

            G.loc[need, correspondant_commodities] = 1

        return G

    @property
    def J(self):
        J = pd.DataFrame(
            data=0,
            index=pd.Index(self.data_factory.technologies, name=INDEX.technology),
            columns=pd.Index(self.data_factory.activities, name=INDEX.activity),
        )

        for technology in self.data_factory.technologies:
            correspondant_activities = self.data_factory.find_activity_by_technology(
                technology
            )

            J.loc[technology, correspondant_activities] = 1

        return J

    @property
    def s(self):
        return pd.DataFrame(
            data=np.identity(len(self.data_factory.activities)),
            index=pd.MultiIndex.from_frame(self.data_factory.activity_units),
            columns=pd.MultiIndex.from_frame(self.data_factory.commodity_units),
        )

    @property
    def u(self):
        return pd.DataFrame(
            data=DEFAULT.u,
            index=pd.MultiIndex.from_frame(self.data_factory.need_units),
            columns=pd.MultiIndex.from_frame(self.data_factory.activity_units),
        )

    @property
    def v(self):
        return pd.DataFrame(
            data=DEFAULT.v,
            index=pd.MultiIndex.from_frame(self.data_factory._factors),
            columns=pd.MultiIndex.from_frame(self.data_factory.activity_units),
        )

    @property
    def e(self):
        return pd.DataFrame(
            data=DEFAULT.e,
            index=pd.MultiIndex.from_frame(self.data_factory._satellite),
            columns=pd.MultiIndex.from_frame(self.data_factory.activity_units),
        )

    @property
    def k(self):
        return pd.DataFrame(
            data=DEFAULT.k,
            index=pd.MultiIndex.from_frame(self.data_factory.technology_units),
            columns=pd.MultiIndex.from_frame(self.data_factory._information),
        )

    @property
    def Y(self):
        return pd.DataFrame(
            data=DEFAULT.Y,
            index=pd.MultiIndex.from_frame(self.data_factory.need_units),
            columns=pd.Index(self.data_factory.operation_time),
        )

    @property
    def Y_coat(self):
        return deepcopy(self.Y)

    @property
    def A(self):
        index = []
        for ii,info in self.data_factory.activity_units.iterrows():
            activity = info[HEADERS.activity_name]
            unit = info[HEADERS.activity_unit]
            if activity in self.data_factory._storage_activities[HEADERS.activity_name].tolist():
                extra_unit = self.data_factory._storage_activities.set_index(HEADERS.activity_name).loc[activity,HEADERS.technology_unit]

            elif activity in self.data_factory._solar_activities[HEADERS.activity_name].tolist():
                extra_unit = self.data_factory._solar_activities.set_index(HEADERS.activity_name).loc[activity,HEADERS.technology_unit]

            elif activity in self.data_factory._technology_need[HEADERS.activity_name].tolist():
                extra_unit = self.data_factory._technology_need.set_index(HEADERS.activity_name).loc[activity,HEADERS.technology_unit]


            index.append((activity,unit,extra_unit))

        print(index)
        return pd.DataFrame(
            data=DEFAULT.A,
            index=pd.MultiIndex.from_tuples(index, names=(INDEX.activity,'Activity unit',"Technology unit")),
            columns=pd.Index(self.data_factory.operation_time),
        )

    @property
    def A_t(self):
        return pd.DataFrame(
            data=DEFAULT.A_t,
            index=pd.MultiIndex.from_frame(self.data_factory.technology_units),
            columns=pd.Index(self.data_factory.operation_time),
        )






# %%
