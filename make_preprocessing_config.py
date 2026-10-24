"""Build the BRFSS preprocessing config from column names and coding rules.

Default rules from the BRFSS 2015 codebook (LLCP 2015 Codebook Report, CDC):
7/9, 77/99 and 777/999 are "don't know"/"refused" (missing); 88/888 and 555
mean "none"/"never" (zero). Overrides list the columns whose codebook entry
differs; every column was checked against the codebook. The dataset stores
calculated variables with implied decimals applied (e.g. 99000 -> 99).
Column maxima are read from the training features only; labels are not used.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data import DATA_PATH, ROOT, load_data

ADMIN = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
         "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT", "CTELNUM1",
         "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT",
         "QSTVER", "QSTLANG", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT",
         "_DUALUSE", "_DUALCOR", "_LLCPWT"]
# Raw answers that BRFSS also provides as cleaned calculated variables.
RAW_DUPLICATES = ["WEIGHT2", "HEIGHT3", "HTIN4", "ALCDAY5", "FRUITJU1", "FRUIT1", "FVBEANS",
                  "FVGREEN", "FVORANG", "VEGETAB1", "EXEROFT1", "EXEROFT2", "EXERHMM1",
                  "EXERHMM2", "STRENGTH", "BLDSUGAR", "FEETCHK2", "LONGWTCH", "FLSHTMY2",
                  "HIVTSTD3", "_CRACE1", "_CPRACE", "_PRACE1", "_MRACE1", "_RACEG21",
                  "_RACEGR3", "_RACE_G1", "_AGE65YR", "_AGE_G"]
# Nominal activity types with ~75 levels, and a multiple-response code.
NOMINAL_DROPPED = ["EXRACT11", "EXRACT21", "PCDMDECN"]
CATEGORICAL = ["_STATE", "MARITAL", "EMPLOY1", "_RACE", "RENTHOM1", "BPHIGH4", "DIABETE3"]
DAILY_FOOD = ["FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_"]
MISSING_OVERRIDES = {"EMPLOY1": [9], "_RACE": [9], "INCOME2": [77, 99], "_AGEG5YR": [14],
                     "DIABAGE2": [98, 99], "ASTHMAGE": [98, 99], "ASERVIST": [98],
                     "ASDRVIST": [98], "ASRCHKUP": [98, 99], "CHILDREN": [99],
                     "SCNTWRK1": [97, 99], "SCNTLWK1": [97, 99], "SEATBELT": [7, 8, 9],
                     "VIDFCLT2": [7, 8], "VIREDIF3": [7], "VIINSUR2": [7, 8, 9],
                     "VICTRCT4": [7], "VIGLUMA2": [7], "VIMACDG2": [7], "ASATTACK": [7],
                     "RDUCHART": [7, 9], "ASNOSLEP": [7], "_DRNKWEK": [99900],
                     "DROCDY3_": [900], "MAXVO2_": [999], "FC60_": [999],
                     "PAFREQ1_": [99], "PAFREQ2_": [99], "STRFREQ_": [99],
                     **{name: [] for name in DAILY_FOOD}}
NO_CODES = ["_AGE80", "WTKG3", "HTM4", "_BMI5", "NUMADULT", "NUMMEN", "NUMWOMEN"]
ZERO_OVERRIDES = {"CHILDREN": [88], "ASERVIST": [88], "ASDRVIST": [88], "DIABAGE2": [],
                  "ASTHMAGE": [], "SCNTWRK1": [98], "SCNTLWK1": [98], "CHKHEMO3": [88, 98],
                  "ASNOSLEP": [8], "ASTHMED3": [8], "ASINHALR": [8], "MAXVO2_": [],
                  "FC60_": [], "PAFREQ1_": [], "PAFREQ2_": [], "STRFREQ_": [],
                  **{name: [] for name in DAILY_FOOD}}


def column_names(data_dir):
    with (Path(data_dir) / "x_train.csv").open(encoding="utf-8") as stream:
        return stream.readline().strip().split(",")[1:]


def build_config(names, maxima):
    drop = set(ADMIN + RAW_DUPLICATES + NOMINAL_DROPPED)
    missing, zero = {}, {}
    for j, (name, top) in enumerate(zip(names, maxima)):
        if name in drop or name in NO_CODES or np.isnan(top):
            codes, zeros = [], []
        elif top == 9:
            codes, zeros = ([9] if name.startswith("_") else [7, 9]), []
        elif top == 99:
            codes, zeros = [77, 99], [88]
        elif top == 999:
            codes, zeros = [777, 999], [888, 555]
        else:
            codes, zeros = [], []
        codes = MISSING_OVERRIDES.get(name, codes)
        zeros = ZERO_OVERRIDES.get(name, zeros)
        if codes:
            missing[str(j)] = codes
        if zeros:
            zero[str(j)] = zeros
    index = {name: j for j, name in enumerate(names)}
    return {"missing_codes": missing, "zero_codes": zero,
            "categorical_columns": [index[n] for n in CATEGORICAL],
            "drop_columns": sorted(index[n] for n in drop),
            "add_missing_indicators": False, "degree": 1,
            "standardize": True, "add_bias": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "config" / "preprocessing_brfss.json")
    args = parser.parse_args(argv)
    names = column_names(args.data_dir)
    x_train = load_data(args.data_dir)[0]
    with np.errstate(all="ignore"):
        maxima = np.nanmax(np.where(np.isnan(x_train), -np.inf, x_train), axis=0)
    maxima[np.isinf(maxima)] = np.nan
    config = build_config(names, maxima)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=1)
        stream.write("\n")
    print(f"{len(config['drop_columns'])} dropped, {len(config['categorical_columns'])} categorical, "
          f"{len(config['missing_codes'])} with missing codes, {len(config['zero_codes'])} with zero codes")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
