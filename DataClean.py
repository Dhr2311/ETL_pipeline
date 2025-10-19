import pandas as pd
import pandas as pd
import numpy as np
import re


def simplyfy_cols(df):
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()                 
        .str.lower()               
        .str.replace(r'[\s_]+', '', regex=True) 
    )
    return out


def clean_diagnosis(df):
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    out.dropna(subset=['encounterid'],inplace=True)

    # --- normalize booleans ---
    def to_bool(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().lower()
        if s in {'true','t','1','yes','y'}:  return True
        if s in {'false','f','0','no','n','none'}: return False
        return np.nan
    out['isprimary'] = out['isprimary'].apply(to_bool)


    out['recorded_at_utc'] = pd.to_datetime(out['recordedat'], errors='coerce', utc=True)

    out['recorded_date'] = out['recorded_at_utc'].dt.date
    now_utc = pd.Timestamp.now(tz='UTC')
    out['is_future'] = out['recorded_at_utc'] > now_utc

    
    out = out.drop_duplicates(subset=['encounterid','code','recorded_at_utc'], keep='last')


    cols = ['encounterid','code','isprimary','recorded_at_utc','recorded_date','is_future']
    other = [c for c in out.columns if c not in cols]
    return out[cols + other]



def clean_encounters(df2):
    row = df2.iloc[9] 
    df2.iloc[9] = row["encounterid"].split(';')[:-1]
    df2.drop(5, inplace=True)#could make this a condition based like if encounter id in encounter id column etc
    return df2




def clean_patients(df3):
    df3 = df3.drop_duplicates(subset=['patientid'], keep='first')

    # 3. Normalize date formats (auto-detect different date styles)
    df3['dob'] = pd.to_datetime(df3['dob'], errors='coerce', dayfirst=True)

    # 4. Convert height → centimeters
    def to_cm(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower().replace("”", '"').replace("′", "'")

        # 1) cm
        if 'cm' in s:
            num = re.findall(r'[\d.]+', s)
            return float(num[0]) if num else np.nan

        # 2) ft + in (handles 5ft6in, 5 ft 6 in, 5'6", etc.)
        match = re.match(r"(?:(\d+)\s*(?:ft|'))\s*(\d+)?", s)
        if match:
            ft = float(match.group(1))
            inch = float(match.group(2)) if match.group(2) else 0
            return round((ft * 12 + inch) * 2.54, 1)

        # 3) only inches
        if 'in' in s or '"' in s:
            num = re.findall(r'[\d.]+', s)
            if num:
                return round(float(num[0]) * 2.54, 1)

        # 4) meters (e.g. 1.7m)
        if 'm' in s:
            num = re.findall(r'[\d.]+', s)
            if num:
                return round(float(num[0]) * 100, 1)

        # 5) bare number — assume centimeters
        num = re.findall(r'[\d.]+', s)
        return float(num[0]) if num else np.nan
    
    df3['height_cm'] = df3['height'].apply(to_cm)

    # 5. Convert weight → kilograms
    def to_kg(w):
        if pd.isna(w): return np.nan
        w = str(w).lower().strip()
        if 'kg' in w:
            return float(re.findall(r'[\d\.]+', w)[0])
        if 'lb' in w:
            pounds = float(re.findall(r'[\d\.]+', w)[0])
            return round(pounds * 0.453592, 1)
        return float(re.findall(r'[\d\.]+', w)[0]) if re.findall(r'[\d\.]+', w) else np.nan

    df3['weight_kg'] = df3['weight'].apply(to_kg)

    # 6. Clean up sex codes (F/M/O/U → keep but uppercase)
    df3['sex'] = df3['sex'].str.upper().replace({'U': np.nan})

    # 7. Drop old messy columns
    df3 = df3.drop(columns=['height', 'weight'])

    return df3


def normalize_key(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.strip()         # <-- THIS fixes the trailing/leading spaces
         .str.upper()
    )




def merge(df1,df2,df3):

    if "encounterid" in df1: df1["encounterid"] = normalize_key(df1["encounterid"])
    if "encounterid" in df2:       df2["encounterid"]       = normalize_key(df2["encounterid"])
    if "patientid"   in df2:       df2["patientid"]         = normalize_key(df2["patientid"])
    if "patientid"   in df3:       df3["patientid"]         = normalize_key(df3["patientid"])

    if "encounterid" in df2:
        df2 = df2.sort_index().drop_duplicates(subset=["encounterid"], keep="last")
    if "patientid" in df3:
        df3 = df3.sort_index().drop_duplicates(subset=["patientid"], keep="last")


    merged = (
        df1
        .merge(df2, on="encounterid", how="left", suffixes=("", "_df2"), indicator=True, validate="many_to_one")
    )
    
    merged = merged.drop(columns=["_merge"])

    master = (
        merged
        .merge(df3, on="patientid", how="left", suffixes=("", "_df3"), indicator=True, validate="many_to_one")
    )
    
    master = master.drop(columns=["_merge"])

    return master


def main (path):
    path_diagnosis = str(path)+"/"+"diagnoses.xml"
    path_encounters = str(path)+"/"+"encounters.csv"
    path_patients = str(path)+"/"+"patients.csv"
    print(path_diagnosis,path_encounters, path_patients)
    df1 = pd.read_xml(path_diagnosis)
    df1 = simplyfy_cols(df1)
    df2 = simplyfy_cols(pd.read_csv(path_encounters))
    df3 = simplyfy_cols(pd.read_csv(path_patients))
    df1 = clean_diagnosis(df1)
    df2 = clean_encounters(df2)
    df3 = clean_patients(df3)
    master = merge (df1,df2,df3)
    master['recorded_at_utc'] = master['recorded_at_utc'].dt.tz_localize(None)
    master.to_excel("data/output.xlsx")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
