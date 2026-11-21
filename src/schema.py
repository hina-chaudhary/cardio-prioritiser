import pandera as pa
from pandera import Column, Check

binary = Check.isin([0, 1])

schema = pa.DataFrameSchema(
    {
        "age": Column(float, Check.in_range(40, 95)),
        "anaemia": Column(int, binary),
        "creatinine_phosphokinase": Column(int, Check.ge(0)),
        "diabetes": Column(int, binary),
        "ejection_fraction": Column(int, Check.in_range(10, 80)),
        "high_blood_pressure": Column(int, binary),
        "platelets": Column(float, Check.gt(0)),
        "serum_creatinine": Column(float, Check.gt(0)),
        "serum_sodium": Column(int, Check.in_range(100, 150)),
        "sex": Column(int, binary),
        "smoking": Column(int, binary),
        "time": Column(int, Check.ge(0)),          # follow-up days
        "DEATH_EVENT": Column(int, binary),        # 1 = event, 0 = censored
    },
    strict=True,        # fail on unexpected columns
    coerce=True,
    checks=Check(lambda df: len(df) == 299, error="Expected 299 rows"),
)