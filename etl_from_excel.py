import os, re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
import os
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env in the same folder


DATABASE_URL = os.environ["DATABASE_URL"]
url = make_url(DATABASE_URL)


load_dotenv()
ENGINE_URL = os.environ["DATABASE_URL"]
EXCEL_PATH = os.environ.get("EXCEL_PATH", "./data/output.xlsx")

engine = create_engine(ENGINE_URL, future=True)

# ----------------- helpers -----------------
def tidy_cols(df):
    out = df.copy()
    out.columns = (out.columns
        .str.strip().str.lower()
        .str.replace(r'[\s_]+', '', regex=True))
    return out

def to_bool(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"true","t","1","yes","y"}:  return True
    if s in {"false","f","0","no","n","none","null","nan"}: return False
    return np.nan

icd10_re = re.compile(r'^[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4})?$')

def clean_types(df):
    out = tidy_cols(df)

    # normalize keys if present
    for c in ("encounterid","patientid","givenname","familyname","sex"):
        if c in out:
            out[c] = out[c].astype("string").str.strip()

    # standardize booleans
    if "isprimary" in out:
        out["isprimary"] = out["isprimary"].apply(to_bool)

    # validate code
    if "code" in out:
        out["code"] = out["code"].astype(str).str.strip().str.upper()
        out["code_valid"] = out["code"].apply(lambda c: bool(icd10_re.match(c)))
    else:
        out["code_valid"] = np.nan

    # recordedat → UTC ts
    if "recordedat" in out:
        out["recorded_at_utc"] = pd.to_datetime(out["recordedat"], errors="coerce", utc=True)
        out["recorded_date"]   = out["recorded_at_utc"].dt.date
        out["is_future"]       = out["recorded_at_utc"] > pd.Timestamp.now(tz="UTC")

    # DoB to date if present
    if "dob" in out:
        out["dob"] = pd.to_datetime(out["dob"], errors="coerce").dt.date

    # numeric vitals if available
    if "height_cm" in out:
        out["height_cm"] = pd.to_numeric(out["height_cm"], errors="coerce")
    if "weight_kg" in out:
        out["weight_kg"] = pd.to_numeric(out["weight_kg"], errors="coerce")

    # flags
    if "encounterid" in out:
        out["missing_encounterid"] = out["encounterid"].isna() | (out["encounterid"] == "")
    if "patientid" in out:
        out["missing_patientid"] = out["patientid"].isna() | (out["patientid"] == "")

    # drop exact dups on strongest key set (if present)
    subset = [c for c in ["encounterid","code","recorded_at_utc"] if c in out]
    if subset:
        out = out.drop_duplicates(subset=subset, keep="last")

    return out

def dq_checks(df):
    issues = []

    def add(idx, reason):
        row = df.loc[idx].to_dict()
        issues.append({
            "row_idx": int(idx),
            "reason": reason,
            "encounterid": row.get("encounterid"),
            "patientid": row.get("patientid"),
            "code": row.get("code"),
            "recordedat": row.get("recordedat"),
            "sourcefile": row.get("sourcefile"),
        })

    if "missing_encounterid" in df:
        for i in df.index[df["missing_encounterid"] == True]: add(i, "missing_encounterid")
    if "missing_patientid" in df:
        for i in df.index[df["missing_patientid"] == True]: add(i, "missing_patientid")
    if "code_valid" in df:
        for i in df.index[df["code_valid"] == False]: add(i, "invalid_icd_code")
    if "recorded_at_utc" in df:
        for i in df.index[df["recorded_at_utc"].isna()]: add(i, "unparsed_recorded_at")
        for i in df.index[df["is_future"] == True]: add(i, "future_recorded_at")
    # plausible ranges
    if "height_cm" in df:
        bad = df.index[(df["height_cm"] < 120) | (df["height_cm"] > 230)]
        for i in bad: add(i, "implausible_height")
    if "weight_kg" in df:
        bad = df.index[(df["weight_kg"] < 25) | (df["weight_kg"] > 300)]
        for i in bad: add(i, "implausible_weight")

    dq = pd.DataFrame(issues, columns=[
        "row_idx","reason","encounterid","patientid","code","recordedat","sourcefile"
    ])
    return dq

def impute_missing_weight(df):
    out = df.copy()
    if not {"height_cm","weight_kg"}.issubset(out.columns):
        out["weight_pred"] = np.nan
        out["imputation_confidence"] = np.nan
        return out, pd.DataFrame()

    train = out.dropna(subset=["height_cm","weight_kg"])
    if len(train) >= 10:
        X = train[["height_cm"]]
        y = train["weight_kg"]
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        conf = "High" if r2 > 0.7 else ("Medium" if r2 > 0.4 else "Low")

        miss_mask = out["weight_kg"].isna() & out["height_cm"].notna()
        out.loc[miss_mask, "weight_pred"] = model.predict(out.loc[miss_mask, ["height_cm"]])
        out.loc[miss_mask, "imputation_confidence"] = conf
    else:
        # fallback: BMI-based using median BMI from observed rows
        train = out.dropna(subset=["height_cm","weight_kg"])
        if len(train) > 0:
            bmi = train["weight_kg"] / (train["height_cm"]/100.0)**2
            med_bmi = float(bmi.median())
            miss_mask = out["weight_kg"].isna() & out["height_cm"].notna()
            out.loc[miss_mask, "weight_pred"] = med_bmi * (out.loc[miss_mask, "height_cm"]/100.0)**2
            out.loc[miss_mask, "imputation_confidence"] = "Heuristic"
        else:
            out["weight_pred"] = np.nan
            out["imputation_confidence"] = np.nan

    # log imputed rows
    log = out.loc[out["weight_kg"].isna() & out["weight_pred"].notna(),
                  ["patientid","height_cm","weight_pred","imputation_confidence","sourcefile"]].copy()
    log["reason"] = "Imputed_weight_from_height"
    log = log.reset_index(names="row_idx") if out.index.name or isinstance(out.index, pd.RangeIndex) else log.reset_index().rename(columns={"index":"row_idx"})
    return out, log

# ----------------- ETL -----------------
def main():
    # 1) Load Excel (first sheet or named)
    df = pd.read_excel(EXCEL_PATH)
    df = clean_types(df)

    # 2) Impute missing weight
    df, dq_imputed = impute_missing_weight(df)

    # 3) DQ table
    dq = dq_checks(df)
    dq = pd.concat([dq, dq_imputed.rename(columns={"imputation_confidence":"confidence"})], ignore_index=True, sort=False)

    # 4) Write to Postgres
    with engine.begin() as conn:
        df.to_sql("encounters_master", conn, if_exists="replace", index=False)
        dq.to_sql("dq_issues", conn, if_exists="replace", index=False)

        # indexes for speed
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_enc_master_encounterid ON encounters_master(encounterid);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_enc_master_patientid   ON encounters_master(patientid);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_dq_reason              ON dq_issues(reason);"))
        except Exception as e:
            print("Index creation warning:", e)

    print(f"Loaded {len(df)} rows to encounters_master")
    print(f"Logged {len(dq)} DQ rows to dq_issues")

if __name__ == "__main__":
    main()
