#%% -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:53:11 2021

@author: nigolred
"""
import pandas as pd

# Constants
building = "Scuola Marugj"
sets_path = 'inputs/{}/sets.xlsx'.format(building)

# Data Entry
input_nt = pd.read_excel(sets_path, header=[0,1], sheet_name='n-t')
input_f = pd.read_excel(sets_path, header=[0,1], sheet_name='f')
input_l = pd.read_excel(sets_path, header=[0,1], sheet_name='l')
input_i = pd.read_excel(sets_path, header=[0,1], sheet_name='i')
input_time_operation = pd.read_excel(sets_path, header=None, index_col=[0], sheet_name='qo')
input_time_investment = pd.read_excel(sets_path, header=None, index_col=[0], sheet_name='qi')

# Names of group of variables
Nn = input_nt.columns[0][0]
Nt = input_nt.columns[2][0]
Ny = '{} Type'.format(Nt)
Na = 'Activity'
Nc = 'Commodity'
Nf = input_f.columns[0][0]
Nl = input_l.columns[0][0]
Ni = input_i.columns[0][0]
Nqo = input_time_operation.iloc[2,0]
Nqi = input_time_investment.iloc[2,0]

# Introducing activities (a) and commodities (c)
input_nt[(Na,'Name')] = 0
input_nt[(Na,'Unit')] = 0
input_nt[(Nc,'Name')] = 0
input_nt[(Nc,'Unit')] = 0

# Naming and giving the unit
for ii in input_nt.index:
    input_nt.loc[ii,(Na,'Name')] = 'Exploiting {} for {}'.format(input_nt.loc[ii,(Nt,'Name')],input_nt.loc[ii,('Need','Name')])
    input_nt.loc[ii,(Na,'Unit')] = '{}'.format(input_nt.loc[ii,('Need','Unit')])
    input_nt.loc[ii,(Nc,'Name')] = '{} from {}'.format(input_nt.loc[ii,('Need','Name')],input_nt.loc[ii,(Nt,'Name')])
    input_nt.loc[ii,(Nc,'Unit')] = '{}'.format(input_nt.loc[ii,('Need','Unit')])

# Getting all the needed indeces
n = input_nt.loc[:,'Need'].drop_duplicates() # needs
t = input_nt.loc[:,Nt].drop_duplicates() # technologies
a = input_nt.loc[:,Na].drop_duplicates() # activities
c = input_nt.loc[:,Nc].drop_duplicates() # commodities
bat = t.loc[t.Type=='Storage'].reset_index(drop=True) #subset of storage technologies
sol = t.loc[t.Type=='Solar'].reset_index(drop=True) #subset of solar technologies
f = input_f.loc[:,Nf].drop_duplicates() # factor of production
l = input_l.loc[:,Nl].drop_duplicates() # satellite account
i = input_i.loc[:,Ni].drop_duplicates() # technology information
qo = pd.date_range(start=input_time_operation.loc['Data inizio'].values[0], end=input_time_operation.loc['Data fine'].values[0], freq=input_time_operation.loc['Frequenza'].values[0]) # time step for operational decisions
qi = pd.date_range(start=input_time_investment.loc['Data inizio'].values[0], end=input_time_investment.loc['Data fine'].values[0], freq=input_time_investment.loc['Frequenza'].values[0]) # time step for investment decisions

# Adding the charging activities of the storage technologies
for ii in input_nt.index:
    if input_nt.loc[ii,(Nt,'Type')] == 'Storage':
        a = a.append({'Name': 'Charging {}'.format(input_nt.loc[ii,(Nt,'Name')]),
                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()
        c = c.append({'Name': '{} Charge'.format(input_nt.loc[ii,(Nt,'Name')]),
                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()
        l = l.append({'Name': 'Charge from {} '.format(input_nt.loc[ii,(Nt,'Name')]),
                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()
    if input_nt.loc[ii,(Nt,'Type')] == 'Solar':
        a = a.append({'Name': 'Selling {} surplus'.format(input_nt.loc[ii,(Nt,'Name')]),
                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()
        c = c.append({'Name': 'Sold surplus from {}'.format(input_nt.loc[ii,(Nt,'Name')]),
                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()


#%% Building the needed matrices that do not need any interaction with the user
# G, the matrix that connects needs and commodities
n_index = pd.MultiIndex.from_product([list(n.loc[:,'Name'])], names=['Need'])
c_index = pd.MultiIndex.from_product([list(c.loc[:,'Name'])], names=[Nc])
G = pd.DataFrame(0, index=n_index, columns=c_index)
for ni in n.loc[:,'Name']:
    for ci in c.loc[:,'Name']:
         if ci.find(ni)!=-1: G.loc[ni,ci] = 1

# J, the matrix that technologies and activity
t_index = pd.MultiIndex.from_product([list(t.loc[:,'Name'])], names=[Nt])
bat_index = pd.MultiIndex.from_product([list(bat.loc[:,'Name'])], names=['Storage'])
sol_index = pd.MultiIndex.from_product([list(sol.loc[:,'Name'])], names=['Solar'])
a_index = pd.MultiIndex.from_product([list(a.loc[:,'Name'])], names=[Na])
J = pd.DataFrame(0, index=t_index, columns=a_index)
for ti in t.loc[:,'Name']:
    for ai in a.loc[:,'Name']:
         if ai.find(ti)!=-1: J.loc[ti,ai] = 1

# s, the market-share matrix which is by costruction an identity matrix
s = pd.DataFrame(0, index=a_index, columns=c_index)
for ai in range(len(a)):
    for ci in range(len(c)):
        if ai==ci: s.iloc[ai,ci] = 1

i_index = pd.MultiIndex.from_product([list(i.loc[:,'Name'])], names=[Ni])
f_index = pd.MultiIndex.from_product([list(f.loc[:,'Name'])], names=[Nf])
l_index = pd.MultiIndex.from_product([list(l.loc[:,'Name'])], names=[Nl])

#%% Building the needed matrices that need interaction with the user. U referes to a user-friendly version of the indeces
#u, matrix of specific use of needs by activity which required user data entry (or look-up on a dedicated database)
nU_index = pd.MultiIndex.from_arrays([list(n.loc[:,'Name']),list(n.loc[:,'Unit'])], names=[Nn,'Unit'])
aU_index = pd.MultiIndex.from_arrays([list(a.loc[:,'Name']),list(a.loc[:,'Unit'])], names=[Na,'Unit'])
# u = pd.DataFrame(0, index=nU_index, columns=aU_index).to_excel('{}/u{}.xlsx'.format(pr_fld,ver)) # Writing indexing with explicit units

# v (and v_), matrix of specific use of factor of production by activity (and time-step) which required user data entry (or look-up on a dedicated database)
fU_index = pd.MultiIndex.from_arrays([list(f.loc[:,'Name']),list(f.loc[:,'Unit'])], names=[Nf,'Unit'])
aqo_index = pd.MultiIndex.from_product([list(a.loc[:,'Name']),qo], names=[Na,input_time_operation.loc['Frequenza'].values[0]])
aqoU_index = pd.MultiIndex.from_product([list('{} [{}]'.format(first, second) for first, second in zip(list(a.loc[:,'Name']), list(a.loc[:,'Unit']))),qo], names=[Na,input_time_operation.loc['Frequenza'].values[0]])
# v_ = pd.DataFrame(0, index=fU_index, columns=aqoU_index).to_excel('{}/v_.xlsx'.format(pr_fld)) # Writing indexing with explicit units
# v = pd.DataFrame(0, index=fU_index, columns=aU_index).to_excel('{}/v{}.xlsx'.format(pr_fld,ver)) # Writing indexing with explicit units

# e (and e_), matrix of specific use of satellite account by activity (and time-step) which required user data entry (or look-up on a dedicated database)
lU_index = pd.MultiIndex.from_arrays([list(l.loc[:,'Name']),list(l.loc[:,'Unit'])], names=[Nl,'Unit'])
# e_ = pd.DataFrame(0, index=lU_index, columns=aqoU_index).to_excel('{}/e_.xlsx'.format(pr_fld)) # Writing indexing with explicit units
# e = pd.DataFrame(0, index=lU_index, columns=aU_index).to_excel('{}/e{}.xlsx'.format(pr_fld,ver)) # Writing indexing with explicit units

# k (and k_), matrix of specific installment cost by technology
kU_index = pd.MultiIndex.from_arrays([list(t.loc[:,'Name']),list(t.loc[:,'Unit'])], names=[Nt,'Unit'])
iU_index = pd.MultiIndex.from_arrays([list(i.loc[:,'Name']),list(i.loc[:,'Unit'])], names=[Na,'Unit'])
iqi_index = pd.MultiIndex.from_product([list(i.loc[:,'Name']),qi], names=[Ni,input_time_investment.loc['Frequenza'].values[0]])
iqiU_index = pd.MultiIndex.from_product([list('{} [{}]'.format(first, second) for first, second in zip(list(i.loc[:,'Name']), list(i.loc[:,'Unit']))),qi], names=[Ni,input_time_investment.loc['Frequenza'].values[0]])
# k = pd.DataFrame(0, index=kU_index, columns=iU_index).to_excel('{}/k{}.xlsx'.format(pr_fld,ver))
# k_ = pd.DataFrame(0, index=kU_index, columns=iqi_index).to_excel('{}/k_.xlsx'.format(pr_fld))

# Y, matrix of final demands
# Y = pd.DataFrame(0, index=nU_index, columns=qo).to_excel('{}/Y{}.xlsx'.format(pr_fld,ver))

# An index that can help the user filling A
tec_act = []
act_nee = []
act_tec = []

for ai in a.loc[:,'Name']:
    if ai in list(input_nt.loc[:,(Na,'Name')]):
        tec_act.append(input_nt.set_index((Na,'Name')).loc[ai,(Nt,'Unit')])
        act_tec.append(input_nt.set_index((Na,'Name')).loc[ai,(Nt,'Name')])
        act_nee.append(input_nt.set_index((Na,'Name')).loc[ai,(Nn,'Name')])
    else:
        for bi in bat.loc[:,'Name']:
            if ai.find(bi)!=-1:
                act_nee.append('Charge for {}'.format(bi))
                if type(input_nt.set_index((Nt,'Name')).loc[bi,(Nt,'Unit')])==str:
                    tec_act.append(input_nt.set_index((Nt,'Name')).loc[bi,(Nt,'Unit')])
                    act_tec.append(bi)
                else:
                    tec_act.append(input_nt.set_index((Nt,'Name')).loc[bi,(Nt,'Unit')][0])
                    act_tec.append(bi)
        for si in sol.loc[:,'Name']:
            if ai.find(si)!=-1:
                act_nee.append('Surplus from {}'.format(si))
                if type(input_nt.set_index((Nt,'Name')).loc[bi,(Nt,'Unit')])==str:
                    tec_act.append(input_nt.set_index((Nt,'Name')).loc[si,(Nt,'Unit')])
                    act_tec.append(si)
                else:
                    tec_act.append(input_nt.set_index((Nt,'Name')).loc[si,(Nt,'Unit')][0])
                    act_tec.append(si)

atU_index = pd.MultiIndex.from_arrays([list(a.loc[:,'Name']),list(a.loc[:,'Unit']),tec_act], names=['Activity Name','Activity Unit','Technology Unit'])
# A, matrix of availability of activities
# A = pd.DataFrame(big, index=atU_index, columns=qo).to_excel('{}/A{}.xlsx'.format(pr_fld,ver))
# At = pd.DataFrame(big, index=kU_index, columns=qo).to_excel('{}/At{}.xlsx'.format(pr_fld,ver))
colors = pd.DataFrame()

#%% Reading-back and re-indexing the matrices filled by the user
u = pd.DataFrame(pd.read_excel('inputs/{}/u.xlsx'.format(building), index_col=[0,1], header=[2]).values, index=n_index, columns=a_index)
A = pd.DataFrame(pd.read_excel('inputs/{}/A.xlsx'.format(building), index_col=[0,1,2], header=[0]).values, index=a_index, columns=qo)
At = pd.DataFrame(pd.read_excel('inputs/{}/A_t.xlsx'.format(building), index_col=[0,1], header=[0]).values, index=t_index, columns=qo)
k = pd.DataFrame(pd.read_excel('inputs/{}/k.xlsx'.format(building), index_col=[0,1], header=[2]).values, index=t_index, columns=i_index)
# k_ = pd.DataFrame(pd.read_excel('{}/k_.xlsx'.format(pj_fld), index_col=[0,1], header=[2]).values, index=t_index, columns=iqi_index)
e = pd.DataFrame(pd.read_excel('inputs/{}/e.xlsx'.format(building), index_col=[0,1], header=[2]).values, index=l_index, columns=a_index)
# e_ = pd.DataFrame(pd.read_excel('{}/e_.xlsx'.format(pj_fld), index_col=[0,1], header=[2]).values, index=l_index, columns=aqo_index)
v = pd.DataFrame(pd.read_excel('inputs/{}/v.xlsx'.format(building), index_col=[0,1], header=[2]).values, index=f_index, columns=a_index)
# v_ = pd.DataFrame(pd.read_excel('{}/v_.xlsx'.format(pj_fld), index_col=[0,1], header=[2]).values, index=f_index, columns=aqo_index)
Y = pd.DataFrame(pd.read_excel('inputs/{}/Y.xlsx'.format(building), index_col=[0,1], header=[0]).values, index=n_index, columns=qo)
Y_coat = pd.DataFrame(pd.read_excel('inputs/{}/Y_coat.xlsx'.format(building), index_col=[0,1], header=[0]).values, index=n_index, columns=qo)
Y_led = pd.DataFrame(pd.read_excel('inputs/{}/Y_led.xlsx'.format(building), index_col=[0,1], header=[0]).values, index=n_index, columns=qo)

#%% Building the endogenous matrices
D = pd.DataFrame(1, index=t_index, columns=['Installed units']) # Number of installed units
D_ = pd.DataFrame(1, index=t_index, columns=qi) # Number of installed units in different investment time-slices
C = pd.DataFrame(1, index=t_index, columns=['Installed capacity']) # Installed capacity
C_ = pd.DataFrame(1, index=t_index, columns=qi) # Installed capacity in different investment time-slices
SoC = pd.DataFrame(1, index=bat_index, columns=qo) # State of charge of every battery technology in every operational time-slice
X = pd.DataFrame(1, index=a_index, columns=qo) # Production of commodities in every operational time-slice

#%% Optimization problem
import cvxpy as cv
import numpy as np

# Reading the missing useful information for running the model
add_par = pd.read_excel(sets_path, sheet_name='additional parameters', header=[0], index_col=[0,1])
dr = add_par.loc['Discount rate','Value'][0]
cp = add_par.loc['CO2 price','Value'][0]

# Extracting slices of exogenous parameters
k_cost = k.loc[:,'Costo di installazione'].values
k_co2 = k.loc[:,'Carbon footprint'].values
e_co2 = e.loc['CO2',:].values
v_cost = v.loc['Costi operativi',:].values
stlv = k.loc[bat.loc[:,'Name'],'Capacità batteria'] # Upper limit of storage technologies
Tri = np.triu(np.ones((len(qo),len(qo)))) # Triangular matrix for SoC
n_st = len(bat) # Number of storage technologies
grid_pos = list(t.loc[:,'Name']).index('Grid') # Position of grid
PVmax_pos = list(t.loc[:,'Name']).index('PV') # Position of PV
PV_pos = list(a.loc[:,'Name']).index('Exploiting PV for Electricity') # Position of PV
PV_limit = list(t.loc[:,'Name']).index('PV')#Position PV
Coat_pos = list(t.loc[:,'Name']).index('Coat') #Position of Coat
LED_pos = list(t.loc[:,'Name']).index('LED bulbs') #Position of Coat
PS_pos = list(t.loc[:,'Name']).index('Psolar')#Position of solar panel
Boiler_pos = list(t.loc[:,'Name']).index('Gas boiler')#Position of Gas boiler
Heatpump_pos = list(t.loc[:,'Name']).index('Heat pump')#Position of Heat pump
Powerwall_pos = list(t.loc[:,'Name']).index('Powerwall')#Position of Powerwall


D = cv.Variable(shape=D.shape, integer=True)
X = cv.Variable(shape=X.shape, nonneg=True)
Obj = cv.Minimize(cv.matmul((cp*k_co2+k_cost).T,D)+8760/len(qo)*sum(cv.sum(cv.sum(cv.matmul(cp*e_co2+v_cost,X),1,keepdims=True),1,keepdims=True)/(1+dr)**i for i in range(len(qi)))) # Miminization of costs


constraints = [D[grid_pos] <= 1, # The grid contract can be 0 or 1
               D[Coat_pos]<= 1,#The Coat can be 0 or 1
               D[PV_limit] <= 15, # The number of PV
               # X[PV_pos] == D[PV_pos]*A.values[PV_pos], # Production of electricity from PV equal to availability
               cv.matmul(G,cv.matmul(s.values.T,X)) == Y.values - D[Coat_pos]*Y_coat.values - D[LED_pos]*Y_led.values + cv.matmul(u.values,X), # Supply of need must be equal to demand and intermediate demand of need
               X <= cv.matmul(cv.diag(cv.matmul(J.T.values,D)),A.values), # Availability of activity
               cv.matmul(J.values,X) <= cv.matmul(cv.diag(D),At.values), # Availability of technology
               cv.matmul(X[-n_st:],Tri) - cv.matmul(cv.matmul(e,X)[-n_st:],Tri) <= cv.matmul(cv.diag(D[t.loc[t.Type=='Storage'].index]),np.repeat(stlv.values.reshape(stlv.shape), len(qo), 1)), # Cannot charge more than capacity of battery
               cv.matmul(X[-n_st:],Tri) - cv.matmul(cv.matmul(e,X)[-n_st:],Tri) >= 0.01*cv.matmul(cv.diag(D[t.loc[t.Type=='Storage'].index]),np.repeat(stlv.values.reshape(stlv.shape), len(qo), 1)) # Cannot run out of more than 1% of battery
               ]

#%% Solve
problem = cv.Problem(Obj, constraints)
problem.solve(verbose=True, solver='GUROBI')

#%%
#%% -*- coding: utf-8 -*-

from mario_u.interface.results import Results
#%%
qo = Y.columns
tqo_index = pd.MultiIndex.from_product([list(t.loc[:,'Name']),qo], names=['Technology','Qo'])

results = {
    "X" : dict(
        value = X.value,
        index = pd.MultiIndex.from_arrays([list(a.loc[:,'Name']),act_tec,act_nee], names=['Activity','Technology','Need']),
        columns = qo,
        ),
    "E" : dict(
        value = e.values @ X.value,
        index = e.index,
        columns = qo,
    ),
    "D" : dict(
        value = D.value,
        index = t_index,
        columns = ["Installed units"],
    ),
    "SOC" : dict(
        value = X.value[-n_st:]@Tri - (e.values@X.value)[-n_st:]@Tri,
        index = bat_index,
        columns = qo,
    ),

    "Y" : Y,

    "investment" : dict(
        value = D.value * k_cost,
        index = t_index,
        columns = ["Investment cost"],
    ),


}
results = Results(problem,results)


Xres = results.results["X"]
Yint = pd.DataFrame((u.values@Xres).values, index=n_index, columns=qo)

tqo_index = pd.MultiIndex.from_product([list(t.loc[:,'Name']),qo], names=['Technology','Qo'])
consumption = pd.DataFrame(0, index=n_index, columns=tqo_index)
impact = pd.DataFrame(0, index=l_index, columns=tqo_index)
costs = pd.DataFrame(0, index=f_index, columns=tqo_index)

for h in qo:
    consumption.loc[:,(slice(None),h)] = u.values@np.diagflat(Xres.loc[:,h].values)@J.T.values
    impact.loc[:,(slice(None),h)] = e.values@np.diagflat(Xres.loc[:,h].values)@J.T.values
    costs.loc[:,(slice(None),h)] = v.values@np.diagflat(Xres.loc[:,h].values)@J.T.values

results.add_results({"consumption":consumption,"impact":impact,"costs":costs})





#%%



#%%
import plotly.graph_objects as go
from mario_u.interface.database import DataFactory
from mario_u.tools.constansts import HEADERS

sets = DataFactory(sets_path)
#%%
class Plots:

    def __init__(self,sets,colors,results):
        self.colors = pd.read_excel(colors,index_col=0,header=0)
        self.results = results.results
        self.sets = sets


    def plot_investment_cost(self,techs,**layout):
        fig = go.Figure()
        unit = self.sets.technology_units.set_index(HEADERS.technology_name)
        for tech in techs:
            fig.add_trace(
                go.Bar(
                    x = [tech + f" [{unit.loc[tech,HEADERS.technology_unit]}]"],
                    y = self.results['investment'].loc[tech,"Investment cost"],
                    marker_color = self.colors.loc[tech,'color'],
                    name = tech + f" [{unit.loc[tech,HEADERS.technology_unit]}]"
                )
            )

        fig.update_layout(showlegend=False,title="Investment cost")
        fig.update_layout(layout)
        fig.show()

        return fig


    def plot_D(self,techs,**layout):
        fig = go.Figure()
        unit = self.sets.technology_units.set_index(HEADERS.technology_name)
        for tech in techs:
            fig.add_trace(
                go.Bar(
                    x = [tech + f" [{unit.loc[tech,HEADERS.technology_unit]}]"],
                    y = self.results['D'].loc[tech,"Installed units"],
                    marker_color = self.colors.loc[tech,'color'],
                    name = tech + f" [{unit.loc[tech,HEADERS.technology_unit]}]"
                )
            )

        fig.update_layout(showlegend=False,title="Capacita Installata")
        fig.update_layout(layout)
        fig.show()

        return fig

    def plot_total_investment(self,**layout):
        investment = self.results['investment']
        total_investment = sum(investment.values)[0]
        shares = investment/total_investment*100

        fig = go.Figure()

        for tech,val in shares.iterrows():
            name = tech[0]
            if sum(val.values) == 0: continue
            fig.add_trace(
                go.Bar(
                    x = ["Investment Share"],
                    y = val.values,
                    name = name,
                    marker_color = self.colors.loc[name,"color"]
                )
            )

        fig.update_layout(
                {
                "barmode" : "relative",
                "title" : f"Total investment cost <br> <sub>{total_investment} Euros</sub>",
                "yaxis":{'title':"[%]"}
                },
            )

        fig.update_layout(**layout)
        fig.show()
        return fig
    def plot_supply_demand(self,need,activities,time_slices,**layout):

        fig = go.Figure()
        counter = [0]
        legends = set()
        for slice,time in time_slices.items():
            final_demand = self.results['Y'].loc[need,time]

            fig.add_trace(
                go.Bar(
                    x = time,
                    y = -final_demand.values.ravel(),
                    name = need,
                    marker_color = self.colors.loc[need,'color']
                )
            )

            for tech in self.results['consumption'].columns.unique(0):
                y = self.results['consumption'].loc[need,(tech,time)].values.ravel()
                if sum(y) != 0:

                    name = "Consumption by " + tech
                    fig.add_trace(
                        go.Bar(
                            x = time,
                            y = - y ,
                            name = name,
                            showlegend=False if name in legends else True,
                            legendgroup = name,
                            marker_color = self.colors.loc[tech,'color']
                        )
                    )
                    legends.add(name)

            for activity in activities:
                fig.add_trace(
                    go.Bar(
                        x = time,
                        y = self.results['X'].loc[activity,time].values.ravel(),
                        name = activity,
                        showlegend=False if activity in legends else True,
                        legendgroup=activity,
                        marker_color = self.colors.loc[activity,'color']
                    )
                )
                legends.add(activity)

            counter.append(len(fig.data))


        steps = []
        for step_index, step in enumerate(
            counter[0:-1]
        ):
            steps.append(
                dict(
                    label=[*time_slices][step_index],
                    method="update",
                    args=[
                        {
                            "visible": [
                                True
                                if counter[step_index] <= i < counter[step_index + 1]
                                else False
                                for i in range(len(fig.data))
                            ]
                        },
                    ],
                )
            )


        fig.update_layout(
                {
                "updatemenus":[dict(active=0, buttons=steps, pad=dict(t=50))],
                "barmode" : "relative"
                },
            )

        fig.update_layout(**layout)
        fig.show()

        return fig


    def plot_operational_costs(self,**layout):
        pass





layout = dict(
    font_family='Palatino Linotype',
    template = 'simple_white',
    font_size=20,
)
plt = Plots(sets,f"inputs/{building}/colors.xlsx",results)

elect_capacity_fig = plt.plot_D(['PV',"Grid"],**layout)
cost = plt.plot_investment_cost(['PV',"Grid"],**layout)
shares = plt.plot_total_investment(**layout)
#%%
t = plt.plot_supply_demand(
    need="Electricity",
    activities=["Exploiting Grid for Electricity","Exploiting PV for Electricity","Exploiting Powerwall for Electricity"],
    time_slices={
        "Spring":sets.operation_time[0:24*7],
        "Summer" : sets.operation_time[24*7:24*7*2],
        # "Autumn" : sets.operation_time[48:72],
        # "Winter" : sets.operation_time[72:],
        }
    )
#%%
# save the results
results.to_excel("results")







# %% Compute and display costs

# Cost_y = 12*Cost.groupby(level=0,axis=1).sum().T
# Inv = Dres * k_cost
# Exp = pd.DataFrame(0, index=Inv.index, columns=['Investment costs [€]','Operative costs [€/y]'])

# for tec in t.Name:
#     Exp.loc[tec,'Investment costs [€]'] = abs(Inv.loc[tec,:].values)
#     Exp.loc[tec,'Operative costs [€/y]'] = abs(Cost_y.loc[tec,:].values)

# Exp_plt = Exp.stack()
# Exp_plt = Exp_plt.reset_index()
# Exp_plt.columns=['tec','step','soldi']

# fig = px.treemap(Exp_plt, path=[px.Constant("tot"),'step','tec'], values='soldi')
# # fig.write_html('{}/Scelta.html'.format(pj_fld))
# # Preparing plotting

# Choice = Dres.loc[Dres.loc[:,'Installed units']!=0]
# drop_needs = []
# n_plt = n.set_index('Name').drop(drop_needs)

# size_plt = [0.9] # first value for Ele
# for i in range(len(n_plt)-1): size_plt.append((1-size_plt[0])/(len(n_plt)-1))

# subt_plt = []
# for j in n_plt.index: subt_plt.append('{} [{}]'.format(j,n_plt.loc[j][0]))

# # Plotting
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go


# # Operational

# fig = make_subplots(rows=len(n_plt), cols=1, subplot_titles=subt_plt, row_heights=size_plt, shared_xaxes='all')

# for ni in list(n_plt.index):
#     if ni == 'Electricity':
#         #for bi in list(SOC.index):
#             #fig.add_trace(go.Scatter(name='State of charge of {}'.format(bi[0]), x=SOC.columns, y=SOC.loc[bi,:].values, marker=dict(color=input_nt.set_index(('Technology','Name')).loc[:,'Plot'].drop_duplicates().loc[bi,'Colors']), line=dict(dash='dash')), col=1, row=list(n_plt.index).index(ni)+1)
#         for ti in sol.Name:
#             if ti in Choice.index.get_level_values(0):
#                 fig.add_trace(go.Bar(name='Surplus of {} sold'.format(ni), x=Xres.loc['Selling {} surplus'.format(ti)].columns, y=Xres.loc['Selling {} surplus'.format(ti)].values[0], marker=dict(color='gold')), col=1, row=list(n_plt.index).index(ni)+1)
#                 fig.add_trace(go.Scatter(name='Solar availability', x=Xres.loc['Selling {} surplus'.format(ti)].columns, y=Dres.loc[sol.Name[0]].values[0][0]*A.loc['Selling {} surplus'.format(ti)].values[0]), col=1, row=list(n_plt.index).index(ni)+1)

#     fig.add_trace(go.Scatter(name='Total demand of {}'.format(ni), x=Y.columns, y=Y.loc[ni,:].values[0]+Yint.loc[ni,:].values[0]), col=1, row=list(n_plt.index).index(ni)+1)

#     for ti in Choice.index.get_level_values(0):
#         # col = input_nt.set_index(('Technology','Name')).loc[:,'Plot'].drop_duplicates().loc[ti,'Colors']
#         try:
#             fig.add_trace(go.Bar(name=ti, x=Xres.columns, y=Xres.loc[(slice(None),ti,ni),:].values[0], marker=dict(color=col)), col=1, row=list(n_plt.index).index(ni)+1)
#         except:
#             pass
#         try:
#             fig.add_trace(go.Bar(name=ti, x=Cons.loc[ni,ti].columns, y=-Cons.loc[ni,ti].values[0], opacity=0.5, marker=dict(color=col)), col=1, row=list(n_plt.index).index(ni)+1)
#         except:
#             pass

#     fig.add_trace(go.Bar(name='Final demand', x=Y.columns, y=-Y.loc[ni,:].values[0], opacity=0.5, marker=dict(color='pink')), col=1, row=list(n_plt.index).index(ni)+1)


# fig.update_layout(barmode='relative',
#                   title='{}. Investimento totale: {}€'.format(str(Choice),(k_cost.T@Dres.values)[0][0]),
#                   showlegend=True,
#                   font_family='Palatino Linotype',
#                   template = 'simple_white',
#                   font_size=20,
#                   xaxis3_rangeslider_visible=True)


# fig.write_html(f'results/{building}/Result_1.html')



# # New plots
# fig = make_subplots(rows=1, cols=1, shared_xaxes='all')

# for ni in ['Electricity','Heating','Transport','Heat','Gas','HSW']:
#     fig = make_subplots(rows=1, cols=1, subplot_titles=[ni,'SoC'], shared_xaxes='all')
#     fig.add_trace(go.Scatter(name='Total demand of {}'.format(ni), x=Y.columns, y=Y.loc[ni,:].values[0]+Yint.loc[ni,:].values[0]), col=1, row=1)
#     for ti in Choice.index.get_level_values(0):
#         col = input_nt.set_index(('Technology','Name')).loc[:,'Plot'].drop_duplicates().loc[ti,'Colors']
#         try:
#             fig.add_trace(go.Bar(name=ti, x=Xres.columns, y=Xres.loc[(slice(None),ti,ni),:].values[0], marker=dict(color=col)), col=1, row=1)
#         except:
#             pass
#         try:
#             fig.add_trace(go.Bar(name=ti, x=Cons.loc[ni,ti].columns, y=-Cons.loc[ni,ti].values[0], opacity=0.5, marker=dict(color=col)), col=1, row=1)
#         except:
#             pass
#     fig.update_layout(barmode='relative',
#                     title='{}. Investimento totale: {}€'.format(str(Choice),(k_cost.T@Dres.values)[0][0]),
#                     showlegend=True,
#                     font_family='Palatino Linotype',
#                     template = 'simple_white',
#                     font_size=10,
#                     xaxis_rangeslider_visible=False)


#     fig.write_html(f'results/{building}/Result_2.html')


# %%

# qo = Y.columns

# Xres_index = pd.MultiIndex.from_arrays([list(a.loc[:,'Name']),act_tec,act_nee], names=['Activity','Technology','Need'])
# Xres = pd.DataFrame(X.value,index=Xres_index, columns=qo)
# Eres = pd.DataFrame(e.values@X.value, index=e.index, columns=qo)
# Dres = pd.DataFrame(D.value,index=t_index, columns=['Installed units'])
# SOC = pd.DataFrame(X.value[-n_st:]@Tri - (e.values@X.value)[-n_st:]@Tri, index=bat_index, columns=qo)
# Yint = pd.DataFrame((u.values@Xres).values, index=n_index, columns=qo)

# tqo_index = pd.MultiIndex.from_product([list(t.loc[:,'Name']),qo], names=['Technology','Qo'])
# Cons = pd.DataFrame(0, index=n_index, columns=tqo_index)
# Impa = pd.DataFrame(0, index=l_index, columns=tqo_index)
# Cost = pd.DataFrame(0, index=f_index, columns=tqo_index)

# for h in qo:
#     Cons.loc[:,(slice(None),h)] = u.values@np.diagflat(Xres.loc[:,h].values)@J.T.values
#     Impa.loc[:,(slice(None),h)] = e.values@np.diagflat(Xres.loc[:,h].values)@J.T.values
#     Cost.loc[:,(slice(None),h)] = v.values@np.diagflat(Xres.loc[:,h].values)@J.T.values

#%% Compute and display costs

# Cost_y = 12*Cost.groupby(level=0,axis=1).sum().T
# Inv = Dres * k_cost
# Exp = pd.DataFrame(0, index=Inv.index, columns=['Investment costs [€]','Operative costs [€/y]'])

# for tec in t.Name:
#     Exp.loc[tec,'Investment costs [€]'] = abs(Inv.loc[tec,:].values)
#     Exp.loc[tec,'Operative costs [€/y]'] = abs(Cost_y.loc[tec,:].values)

# Exp_plt = Exp.stack()
# Exp_plt = Exp_plt.reset_index()
# Exp_plt.columns=['tec','step','soldi']

# fig = px.treemap(Exp_plt, path=[px.Constant("tot"),'step','tec'], values='soldi')
# # fig.write_html('{}/Scelta.html'.format(pj_fld))
# # Preparing plotting

# Choice = Dres.loc[Dres.loc[:,'Installed units']!=0]
# drop_needs = ['Cooking']
# n_plt = n.set_index('Name').drop(drop_needs)

# size_plt = [0.9] # first value for Ele
# for i in range(len(n_plt)-1): size_plt.append((1-size_plt[0])/(len(n_plt)-1))

# subt_plt = []
# for j in n_plt.index: subt_plt.append('{} [{}]'.format(j,n_plt.loc[j][0]))

# # Plotting
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go


# # Operational

# fig = make_subplots(rows=len(n_plt), cols=1, subplot_titles=subt_plt, row_heights=size_plt, shared_xaxes='all')

# for ni in list(n_plt.index):
#     if ni == 'Electricity':
#         #for bi in list(SOC.index):
#             #fig.add_trace(go.Scatter(name='State of charge of {}'.format(bi[0]), x=SOC.columns, y=SOC.loc[bi,:].values, marker=dict(color=input_nt.set_index(('Technology','Name')).loc[:,'Plot'].drop_duplicates().loc[bi,'Colors']), line=dict(dash='dash')), col=1, row=list(n_plt.index).index(ni)+1)
#         for ti in sol.Name:
#             if ti in Choice.index.get_level_values(0):
#                 fig.add_trace(go.Bar(name='Surplus of {} sold'.format(ni), x=Xres.loc['Selling {} surplus'.format(ti)].columns, y=Xres.loc['Selling {} surplus'.format(ti)].values[0], marker=dict(color='gold')), col=1, row=list(n_plt.index).index(ni)+1)
#                 fig.add_trace(go.Scatter(name='Solar availability', x=Xres.loc['Selling {} surplus'.format(ti)].columns, y=Dres.loc[sol.Name[0]].values[0][0]*A.loc['Selling {} surplus'.format(ti)].values[0]), col=1, row=list(n_plt.index).index(ni)+1)

#     fig.add_trace(go.Scatter(name='Total demand of {}'.format(ni), x=Y.columns, y=Y.loc[ni,:].values[0]+Yint.loc[ni,:].values[0]), col=1, row=list(n_plt.index).index(ni)+1)

#     for ti in Choice.index.get_level_values(0):
#         col = input_nt.set_index(('Technology','Name')).loc[:,'Plot'].drop_duplicates().loc[ti,'Colors']
#         try:
#             fig.add_trace(go.Bar(name=ti, x=Xres.columns, y=Xres.loc[(slice(None),ti,ni),:].values[0], marker=dict(color=col)), col=1, row=list(n_plt.index).index(ni)+1)
#         except:
#             pass
#         try:
#             fig.add_trace(go.Bar(name=ti, x=Cons.loc[ni,ti].columns, y=-Cons.loc[ni,ti].values[0], opacity=0.5, marker=dict(color=col)), col=1, row=list(n_plt.index).index(ni)+1)
#         except:
#             pass

#     fig.add_trace(go.Bar(name='Final demand', x=Y.columns, y=-Y.loc[ni,:].values[0], opacity=0.5, marker=dict(color='pink')), col=1, row=list(n_plt.index).index(ni)+1)


# fig.update_layout(barmode='relative',
#                   title='{}. Investimento totale: {}€'.format(str(Choice),(k_cost.T@Dres.values)[0][0]),
#                   showlegend=True,
#                   font_family='Palatino Linotype',
#                   template = 'simple_white',
#                   font_size=20,
#                   xaxis3_rangeslider_visible=True)


# fig.write_html('{}/Result_o.html'.format(pj_fld))



# # New plots
# fig = make_subplots(rows=1, cols=1, shared_xaxes='all')

# for ni in ['Electricity','Heating','Transport','Heat','Gas','HSW']:
#     fig = make_subplots(rows=1, cols=1, subplot_titles=[ni,'SoC'], shared_xaxes='all')
#     fig.add_trace(go.Scatter(name='Total demand of {}'.format(ni), x=Y.columns, y=Y.loc[ni,:].values[0]+Yint.loc[ni,:].values[0]), col=1, row=1)
#     for ti in Choice.index.get_level_values(0):
#         col = input_nt.set_index(('Technology','Name')).loc[:,'Plot'].drop_duplicates().loc[ti,'Colors']
#         try:
#             fig.add_trace(go.Bar(name=ti, x=Xres.columns, y=Xres.loc[(slice(None),ti,ni),:].values[0], marker=dict(color=col)), col=1, row=1)
#         except:
#             pass
#         try:
#             fig.add_trace(go.Bar(name=ti, x=Cons.loc[ni,ti].columns, y=-Cons.loc[ni,ti].values[0], opacity=0.5, marker=dict(color=col)), col=1, row=1)
#         except:
#             pass
#     fig.update_layout(barmode='relative',
#                     title='{}. Investimento totale: {}€'.format(str(Choice),(k_cost.T@Dres.values)[0][0]),
#                     showlegend=True,
#                     font_family='Palatino Linotype',
#                     template = 'simple_white',
#                     font_size=10,
#                     xaxis_rangeslider_visible=False)


#     fig.write_html('{}/{}_o.html'.format(pj_fld,ni))


# # %%
