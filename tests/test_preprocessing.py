import pytest
import numpy as np
import pandas as pd
from src.preprocessing import AmesPreprocessor


@pytest.fixture
def ames_df():
    data = {
        "Electrical": ["SBrkr", np.nan, "FuseA", "SBrkr", "SBrkr", "SBrkr"],
        "Neighborhood": ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Mitchel"],
        "Lot Frontage": [np.nan, 80.0, 68.0, 60.0, 70.0, 75.0],
        "Pool QC": ["Ex", np.nan, np.nan, np.nan, np.nan, np.nan],
        "Alley": [np.nan, np.nan, "Pave", np.nan, np.nan, np.nan],
        "Fence": [np.nan, "MnPrv", np.nan, np.nan, np.nan, np.nan],
        "Misc Feature": [np.nan, np.nan, "Shed", np.nan, np.nan, np.nan],
        "Fireplace Qu": ["Gd", np.nan, "TA", np.nan, np.nan, np.nan],
        "Garage Cars": [2.0, 2.0, np.nan, 3.0, 2.0, 2.0],
        "TotRms AbvGrd": [8, 6, 7, 9, 5, 6],
        "BsmtFin SF 1": [706.0, 978.0, 486.0, np.nan, 0.0, 0.0],
        "Garage Type": ["Attchd", "Attchd", np.nan, "Detchd", "Attchd", "Attchd"],
        "Garage Finish": ["RFn", "Unf", np.nan, "Fin", "Unf", "Unf"],
        "Garage Qual": ["TA", "TA", np.nan, "TA", "TA", "TA"],
        "Garage Cond": ["TA", "TA", np.nan, "TA", "TA", "TA"],
        "Bsmt Qual": ["Gd", "Gd", "TA", np.nan, "TA", "TA"],
        "Bsmt Cond": ["TA", "TA", "TA", np.nan, "TA", "TA"],
        "Bsmt Exposure": ["No", "Gd", "Mn", np.nan, "No", "No"],
        "BsmtFin Type 1": ["GLQ", "ALQ", "BLQ", np.nan, "Rec", "Rec"],
        "BsmtFin Type 2": ["Unf", "Unf", "Unf", np.nan, "Unf", "Unf"],
        "Mas Vnr Type": ["BrkFace", "None", "BrkFace", "Stone", "None", "None"],
        "Garage Area": [548.0, 460.0, np.nan, 608.0, 480.0, 480.0],
        "Garage Yr Blt": [2003.0, 1976.0, np.nan, 2000.0, 1993.0, 1993.0],
        "BsmtFin SF 2": [0.0, 0.0, 0.0, np.nan, 0.0, 0.0],
        "Bsmt Unf SF": [150.0, 284.0, 434.0, np.nan, 0.0, 0.0],
        "Bsmt Full Bath": [1.0, 0.0, 1.0, np.nan, 0.0, 0.0],
        "Bsmt Half Bath": [0.0, 1.0, 0.0, np.nan, 0.0, 0.0],
        "Total Bsmt SF": [856.0, 1262.0, 920.0, np.nan, 0.0, 0.0],
        "Mas Vnr Area": [196.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        "2nd Flr SF": [854.0, 0.0, 866.0, 1000.0, 0.0, 0.0],
        "Year Remod/Add": [2003, 1976, 2002, 2000, 1993, 2000],
        "Year Built": [2003, 1976, 2001, 2000, 1993, 1998],
        "Yr Sold": [2008, 2007, 2008, 2006, 1990, 2009],
        "Gr Liv Area": [1710, 1262, 1786, 2000, 1000, 1200],
        "1st Flr SF": [856, 1262, 920, 1000, 1000, 1200]
    }
    return pd.DataFrame(data)


def test_no_missing_values_after_transform(ames_df):
    preprocessor = AmesPreprocessor()
    out = preprocessor.fit_transform(ames_df)
    assert out.isna().sum().sum() == 0


def test_dropped_columns_are_gone(ames_df):
    preprocessor = AmesPreprocessor()
    out = preprocessor.fit_transform(ames_df)
    expected_dropped = [
        "Pool QC", "Alley", "Fence", "Misc Feature", "Fireplace Qu",
        "Garage Cars", "TotRms AbvGrd", "1st Flr SF", "Garage Yr Blt", "Year Built"
    ]
    for col in expected_dropped:
        assert col not in out.columns


def test_engineered_columns_exist(ames_df):
    preprocessor = AmesPreprocessor()
    out = preprocessor.fit_transform(ames_df)
    engineered = [
        "TotalSF", "HouseAge", "RemodelAge", "HasBasement", 
        "HasGarage", "Has2ndFloor", "IsRemodeled"
    ]
    for col in engineered:
        assert col in out.columns


def test_total_sf_calculation(ames_df):
    preprocessor = AmesPreprocessor()
    out = preprocessor.fit_transform(ames_df)
    expected_totalsf = ames_df["Gr Liv Area"] + ames_df["Total Bsmt SF"].fillna(0)
    pd.testing.assert_series_equal(out["TotalSF"], expected_totalsf, check_names=False)


def test_house_age_never_negative(ames_df):
    preprocessor = AmesPreprocessor()
    out = preprocessor.fit_transform(ames_df)
    assert (out["HouseAge"] >= 0).all()


def test_is_remodeled_correct(ames_df):
    preprocessor = AmesPreprocessor()
    out = preprocessor.fit_transform(ames_df)
    assert out.loc[4, "IsRemodeled"] == 0
    assert out.loc[5, "IsRemodeled"] == 1


def test_no_leakage(ames_df):
    half = len(ames_df) // 2
    train = ames_df.iloc[:half].copy()
    test = ames_df.iloc[half:].copy()
    
    preprocessor = AmesPreprocessor()
    preprocessor.fit(train)
    preprocessor.transform(test)
    
    expected_medians = train.groupby("Neighborhood")["Lot Frontage"].median()
    pd.testing.assert_series_equal(preprocessor._neighborhood_medians, expected_medians)
    