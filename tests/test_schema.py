import pandas as pd
import pandera as pa
import pytest
from src.schema import schema

PATH = "data/heart_failure_clinical_records_dataset.csv"

def test_real_data_passes():
    schema.validate(pd.read_csv(PATH))

def test_no_missing_values():
    assert not pd.read_csv(PATH).isna().any().any()

def test_bad_ejection_fraction_fails():
    df = pd.read_csv(PATH)
    df.loc[0, "ejection_fraction"] = 150
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)

def test_missing_column_fails():
    df = pd.read_csv(PATH).drop(columns=["time"])
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)