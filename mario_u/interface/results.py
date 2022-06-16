#%%
import pandas as pd


class Results:


    def __init__(self,problem,results):

        self.problem = problem

        if self.problem.status in ["infeasible", "unbounded"]:
            raise ValueError("Problem not solved")


        self.results = {}
        self.add_results(results)

    def add_results(self,results):
        for k,v in results.items():
            if isinstance(v,pd.DataFrame):
                self.results[k] = v
            else:
                self.results[k] = pd.DataFrame(v['value'],index=v['index'],columns=v['columns'])

    def to_excel(self,directory,matrices="All"):

        if matrices == "All":
            matrices = [*self.results]

        for matrix in matrices:
            self.results[matrix].to_excel(f"{directory}/{matrix}.xlsx")








# %%
