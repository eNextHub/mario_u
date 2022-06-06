from inspect import Parameter


class Constant(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


SHEETS = Constant(
    technology_need=Constant(sheet_name="n-t", index_col=None, header=[0, 1]),
    factors=Constant(sheet_name="f", index_col=None, header=[0, 1],),
    satellite=Constant(sheet_name="l", index_col=None, header=[0, 1]),
    information=Constant(sheet_name="i", index_col=None, header=[0, 1]),
    investment_time=Constant(sheet_name="qi", index_col=0, header=None),
    operation_time=Constant(sheet_name="qo", index_col=0, header=None,),
)


SHEETS_TO_READ = Constant(
    sets=[
        "technology_need",
        "factors",
        "satellite",
        "information",
        "investment_time",
        "operation_time",
        "storage_activities",
        "solar_activities",
    ],
    main_sets=[
        "technology_need",
        "factors",
        "satellite",
        "information",
        "investment_time",
        "operation_time",
    ],
    inputs=["A", "A_t", "e", "Y", "Y_coat", "u", "v", "k"],
)

TECH_TYPE = Constant(storage="Storage", normal="Normal", solar="Solar",)

HEADERS = Constant(
    need_name=("Need", "Name"),
    need_unit=("Need", "Unit"),
    technology_name=("Technology", "Name"),
    technology_unit=("Technology", "Unit"),
    activity_name=("Activity", "Name"),
    activity_unit=("Activity", "Unit"),
    commodity_name=("Commodity", "Name"),
    commodity_unit=("Commodity", "Unit"),
    technology_type=("Technology", "Type"),
    factor_name=("Factor of production", "Name"),
    factor_unit=("Factor of production", "Unit"),
    satellite_name=("Satellite account", "Name"),
    satellite_unit=("Satellite account", "Unit"),
    information_name=("Technology information", "Name"),
    information_unit=("Technology information", "Unit"),
    start_date="Data inizio",
    end_date="Data fine",
    frequency="Frequenza",
)

INDEX = Constant(
    need=HEADERS.need_name[0],
    commodity=HEADERS.commodity_name[0],
    technology=HEADERS.technology_name[0],
    activity=HEADERS.activity_name[0],
    factor=HEADERS.factor_name[0],
    satellite=HEADERS.satellite_name[0],
    information=HEADERS.information_name[0],
    unit = "Unit"
)


DEFAULT = Constant(u=0, v=0, e=0, k=0, Y=0, A=99999, A_t=99999,)
