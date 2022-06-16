#%%

from copy import deepcopy
from functools import cached_property
from multiprocessing.sharedctypes import Value
import pandas as pd
from sqlalchemy import column

from mario_u.tools.io import SetReader, InputBuilder
from mario_u.tools.constansts import HEADERS, TECH_TYPE, SHEETS_TO_READ


class DataFactory:
    def __init__(self, model_file):

        self._sets = SetReader(file=model_file)
        self._data_builder = InputBuilder(data_factory=self)
        self.matrices = {}

        for sheet in SHEETS_TO_READ.sets:
            setattr(self, "_" + sheet, getattr(self._sets, sheet)())

        self._satellite = self._sets.add_storages_to_satellite(self._satellite,self._storage_activities)

    def find_commodity_by_need(self, need):

        if need not in self.needs:
            raise ValueError("Wrong need")

        return self._technology_need.loc[
            self._technology_need[HEADERS.need_name] == need
        ][HEADERS.commodity_name].tolist()

    def find_activity_by_technology(self, technology):

        if technology not in self.technologies:
            raise ValueError("Wrong tech")

        return self._technology_need.loc[
            self._technology_need[HEADERS.technology_name] == technology
        ][HEADERS.activity_name].tolist()

    def get_input_excels(self,directory,filled_files=False, sheets=SHEETS_TO_READ.inputs):

        if filled_files:
            for sheet in sheets:
                self.matrices[sheet].to_excel(f"{directory}/{sheet}.xlsx")
        else:
            for sheet in sheets:
                getattr(self._data_builder, sheet).to_excel(f"{directory}/{sheet}.xlsx")


    def read_input_excels(self,directory,sheets=SHEETS_TO_READ.inputs):

        for sheet in sheets:
            df = getattr(self._data_builder, sheet)
            index = df.index
            columns = df.columns
            file = pd.read_excel(
                f'{directory}/{sheet}.xlsx',
                index_col = list(range(index.nlevels)),
                header = list(range(columns.nlevels))
            )


            try:
                df.loc[index,columns] = file.loc[index,columns].values

            except KeyError:
                raise ValueError(f"wrong indexing in {sheet}")

            self.matrices[sheet] = df

    def copy(self):

        return deepcopy(self)


    def add_new_sets(self,df,inplace = True):

        if not inplace:
            copy = self.copy()
            copy.add_new_sets(df,inplace=True)
            return copy

        copy = self.copy()

        new_info = copy._sets.add_new_sets(df)

        new_technology_need = copy._technology_need.append(
            new_info["technology_need"]
        )

        new_storage_activities = copy._storage_activities.append(
            new_info["storage_activities"]
        )

        new_solar_activities = copy._solar_activities.append(
            new_info["solar_activities"]
        )

        copy._technology_need = new_technology_need
        copy._storage_activities = new_storage_activities
        copy._solar_activities = new_solar_activities

        new_matrices = {}
        if self.matrices != {} :
            for sheet,df in self.matrices.items():
                new_df = getattr(copy._data_builder,sheet)

                index = df.index
                columns = df.columns

                new_df.loc[index,columns] = df.loc[index,columns].values
                new_matrices[sheet] = new_df

        self.matrices = new_matrices
        self._technology_need = copy._technology_need
        self._storage_activities = copy._storage_activities


    @property
    def activities(self):
        return (
            self._technology_need[HEADERS.activity_name].drop_duplicates().tolist()
            + self._storage_activities[HEADERS.activity_name].drop_duplicates().tolist()
            + self._solar_activities[HEADERS.activity_name].drop_duplicates().tolist()
        )

    @property
    def commodities(self):
        return (
            self._technology_need[HEADERS.commodity_name].drop_duplicates().tolist()
            + self._storage_activities[HEADERS.commodity_name]
            .drop_duplicates()
            .tolist()
            + self._solar_activities[HEADERS.commodity_name]
            .drop_duplicates()
            .tolist()
        )

    @property
    def needs(self):
        return self._technology_need[HEADERS.need_name].drop_duplicates().tolist()

    @property
    def technologies(self):
        return self._technology_need[HEADERS.technology_name].drop_duplicates().tolist()

    @property
    def storages(self):
        return self._technology_need.loc[
            self._technology_need[HEADERS.technology_type] == TECH_TYPE.storage
        ][HEADERS.technology_name].tolist()

    @property
    def solars(self):
        return self._technology_need.loc[
            self._technology_need[HEADERS.technology_type] == TECH_TYPE.storage
        ][HEADERS.technology_name].tolist()

    @cached_property
    def factors(self):
        return self._factors[HEADERS.factor_name].tolist()

    @property
    def satellite(self):
        return self._satellite[HEADERS.satellite_name].tolist()

    @cached_property
    def information(self):
        return self._information[HEADERS.information_name].tolist()

    @cached_property
    def operation_time(self):
        return self._operation_time

    @cached_property
    def investment_time(self):
        return self._investment_time


    @property
    def activity_units(self):
        dfs = [self._technology_need,self._storage_activities,self._solar_activities]
        units = []
        for df in dfs:
            units.append(
                df[[HEADERS.activity_name,HEADERS.activity_unit]].set_index(HEADERS.activity_name)
                )

        return pd.concat(units).reset_index()


    @property
    def commodity_units(self):
        dfs = [self._technology_need,self._storage_activities,self._solar_activities]
        units = []
        for df in dfs:
            units.append(
                df[[HEADERS.commodity_name,HEADERS.commodity_name]].set_index(HEADERS.commodity_name)
                )

        return pd.concat(units).reset_index()

    @property
    def need_units(self):

        df = self._technology_need[[HEADERS.need_name,HEADERS.need_unit]]
        return df.drop_duplicates()


    @property
    def technology_units(self):
        df = self._technology_need[[HEADERS.technology_name,HEADERS.technology_unit]]
        return df.drop_duplicates()


# %%

if __name__ == "__main__":
    model_file = (
        "main.xlsx"
    )
    data = DataFactory(model_file)
    #%%
    data.get_input_excels('test')
    #%%
    data.read_input_excels('test')
    test_frame = pd.read_excel('test.xlsx',header=[0,1])
    # %%n
    data.add_new_sets(test_frame)
    # %%
    data.get_input_excels('test2',filled_files=True)

    # %%


    class Temp:

        def __init__(self,centig):

            self.centig = centig

        @property
        def faren(self):
            return self.centig * 2


    # %%
