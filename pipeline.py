
from __future__ import annotations
import json, os, re, sqlite3, time
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def read_csv_foundation(path: Path):
    """Task 1.1: standard-library CSV reader; no pandas for the initial raw parse."""
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return header, rows

def parse_row(row: dict) -> dict:
    """Convert numeric fields; retain the row and report parse failures."""
    out = dict(row)
    numeric = {"age", "fee_inr", "prior_no_shows", "distance_km"}
    errors = []
    for key in numeric:
        if key in out and str(out[key]).strip() != "":
            try: out[key] = float(str(out[key]).replace(",", "").strip())
            except (TypeError, ValueError): errors.append(f"{key}: {out[key]!r}")
    if "no_show" in out:
        value = str(out["no_show"]).strip().lower()
        if value in {"1","yes","true","y","no-show","no_show"}: out["no_show"] = 1
        elif value in {"0","no","false","n","attended","show"}: out["no_show"] = 0
    out["_parse_errors"] = errors
    return out

def task_1_1():
    path = DATA/"appointments.csv"
    if not path.exists(): raise FileNotFoundError(f"Missing {path}. Add the original data pack.")
    header, rows = read_csv_foundation(path)
    parsed = [parse_row(r) for r in rows]
    rejected = [r for r in parsed if r["_parse_errors"]]
    accepted = [{k:v for k,v in r.items() if k != "_parse_errors"} for r in parsed if not r["_parse_errors"]]
    (OUT/"rejected_rows.json").write_text(json.dumps(rejected, indent=2, default=str), encoding="utf-8")
    print("Header:", header, "\nRaw row count:", len(rows), "\nAccepted:", len(accepted), "\nRejected:", len(rejected))
    return pd.DataFrame(accepted)

def clean_appointments(df: pd.DataFrame):
    """Tasks 1.2–1.3: deterministic cleaning in the order specified by the brief."""
    x = df.copy()
    # Normalize headers while preserving original column semantics.
    x.columns = [str(c).strip().lower() for c in x.columns]
    before = len(x)
    x = x.drop_duplicates().copy()
    if "booking_channel" in x: x["booking_channel"] = x["booking_channel"].astype("string").str.strip().str.title()
    for c in ("appt_date", "appointment_date", "booked_on"):
        if c in x: x[c] = pd.to_datetime(x[c], errors="coerce")
    date_col = "appt_date" if "appt_date" in x else "appointment_date" if "appointment_date" in x else None
    if date_col:
        x = x.dropna(subset=[date_col]).copy()
    # Age validity and missing-value fill.
    if "age" in x:
        x["age"] = pd.to_numeric(x["age"], errors="coerce")
        x.loc[(x.age < 0) | (x.age > 100), "age"] = np.nan
        if x.age.notna().any(): x["age"] = x.age.fillna(x.age.median()).round().astype("Int64")
    # Fill distance and fee with median, then cap distance using Tukey fences.
    capped = {"distance_km": 0, "fee_inr": 0}
    if "distance_km" in x:
        x["distance_km"] = pd.to_numeric(x.distance_km, errors="coerce")
        if x.distance_km.notna().any():
            x["distance_km"] = x.distance_km.fillna(x.distance_km.median())
            q1, q3 = x.distance_km.quantile([.25,.75]); iqr = q3-q1
            lo, hi = q1-1.5*iqr, q3+1.5*iqr
            capped["distance_km"] = int(((x.distance_km < lo)|(x.distance_km > hi)).sum())
            x["distance_km"] = x.distance_km.clip(lo,hi).round(1)
    if "fee_inr" in x:
        x["fee_inr"] = pd.to_numeric(x.fee_inr, errors="coerce")
        if x.fee_inr.notna().any():
            x["fee_inr"] = x.fee_inr.fillna(x.fee_inr.median()).round().astype("Int64")
    if "prior_no_shows" in x:
        x["prior_no_shows"] = pd.to_numeric(x.prior_no_shows, errors="coerce").fillna(0)
    x.to_csv(OUT/"appointments_clean.csv", index=False)
    print({"rows_before": before, "rows_after": len(x), "removed_duplicates": before-len(x), "capped": capped})
    return x

def enrich_with_clinics(df: pd.DataFrame):
    clinics_path = DATA/"clinics.csv"
    if not clinics_path.exists(): raise FileNotFoundError(f"Missing {clinics_path}")
    clinics = pd.read_csv(clinics_path)
    clinics.columns = [c.strip().lower() for c in clinics.columns]
    x = df.copy()
    if "clinic_id" not in x or "clinic_id" not in clinics: raise ValueError("clinic_id missing from appointments or clinics")
    x = x.merge(clinics, on="clinic_id", how="left", suffixes=("", "_clinic"), validate="many_to_one")
    # Parse booking lead time using whichever date column exists.
    appt = "appt_date" if "appt_date" in x else "appointment_date"
    if appt in x and "booked_on" in x:
        x["lead_time_days"] = (pd.to_datetime(x[appt])-pd.to_datetime(x["booked_on"])).dt.days
    x.to_csv(OUT/"clinics_enriched.csv", index=False)
    return x

def task_1_4(df):
    x = df.copy()
    if "booked_on" in x and ("appt_date" in x or "appointment_date" in x):
        dcol = "appt_date" if "appt_date" in x else "appointment_date"
        x["lead_time_days"] = (pd.to_datetime(x[dcol])-pd.to_datetime(x.booked_on)).dt.days
    if "age" in x: x["age_band"] = pd.cut(x.age, bins=[-1,17,35,55,100], labels=["0-17","18-35","36-55","56-100"])
    if "distance_km" in x: x["distance_band"] = pd.cut(x.distance_km, bins=[-0.01,5,15,30,np.inf], labels=["0-5","5-15","15-30","30+"])
    numeric = x.select_dtypes(include=np.number)
    summary = {"numeric_columns": list(numeric.columns), "mean": numeric.mean().to_dict(),
               "std": numeric.std().to_dict(), "min": numeric.min().to_dict(), "max": numeric.max().to_dict()}
    (OUT/"numpy_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return x

def export_sql(df):
    db = sqlite3.connect(OUT/"careloop.db")
    df.to_sql("appointments", db, if_exists="replace", index=False)
    clinics = pd.read_csv(DATA/"clinics.csv")
    clinics.columns = [c.strip().lower() for c in clinics.columns]
    clinics.to_sql("clinics", db, if_exists="replace", index=False)
    db.close()
    df.to_csv(OUT/"appointments_clean.csv", index=False)

def run_part1():
    raw = task_1_1()
    clean = clean_appointments(raw)
    enriched = enrich_with_clinics(clean)
    enriched = task_1_4(enriched)
    export_sql(enriched)
    print("Part 1 complete. Outputs saved in outputs/")

def run_sql_queries():
    db = sqlite3.connect(OUT/"careloop.db")
    sql = (ROOT/"sql"/"02_queries.sql").read_text(encoding="utf-8")
    # Remove SQL line comments before splitting statements.
    import re
    sql = re.sub(r"(?m)^\s*--.*$", "", sql)
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    for i, stmt in enumerate(statements, 1):
        try:
            result = pd.read_sql_query(stmt, db)
            result.to_csv(OUT/f"sql_query_{i:02d}.csv", index=False)
            print(f"Query {i}:\n", result.to_string(index=False))
        except Exception as e: print(f"Query {i} skipped: {e}")
    db.close()

def target_column(df):
    for c in ("no_show","no_show_flag","status"):
        if c in df:
            if c == "status":
                normalized = df[c].astype("string").str.strip().str.lower().str.replace(" ", "_")
                mapped = normalized.map({"no_show": 1, "no-show": 1, "noshow": 1,
                                         "attended": 0, "show": 0, "present": 0})
                return mapped
            return pd.to_numeric(df[c], errors="coerce")
    raise ValueError("No target column found. Expected no_show/no_show_flag/status.")

def model_part3(df):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
    x = df.copy()
    y = target_column(x)
    keep = y.notna()
    x, y = x.loc[keep].copy(), y.loc[keep].astype(int)
    drop = [c for c in x.columns if c in {"appointment_id","patient_id","no_show","no_show_flag","status"} or c.endswith("_id")]
    X = x.drop(columns=drop, errors="ignore")
    # Exclude raw date strings; derive weekday/month when parseable.
    for c in list(X.columns):
        if "date" in c:
            dt = pd.to_datetime(X[c], errors="coerce")
            if dt.notna().any():
                X[c+"_weekday"] = dt.dt.dayofweek
                X[c+"_month"] = dt.dt.month
            X = X.drop(columns=c)
    cat = X.select_dtypes(include=["object","category","string"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num),
        ("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    # Keep original order: first 70% train, final 30% test; no shuffle.
    split = int(len(X)*.70)
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    if ytr.nunique()<2 or yte.nunique()<2: raise ValueError("Chronological split has only one class; inspect supplied data and brief's split rule.")
    candidates = {
      "logistic_regression": LogisticRegression(max_iter=2000,class_weight="balanced"),
      "decision_tree_depth3": DecisionTreeClassifier(max_depth=3,class_weight="balanced",random_state=42),
      "random_forest": RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1)
    }
    rows=[]; fitted={}
    for name, clf in candidates.items():
        pipe=Pipeline([("prep",pre),("model",clf)])
        pipe.fit(Xtr,ytr); fitted[name]=pipe
        p=pipe.predict_proba(Xte)[:,1]
        rows.append({"model":name,"accuracy":accuracy_score(yte,p>=.5),"precision":precision_score(yte,p>=.5,zero_division=0),
                     "recall":recall_score(yte,p>=.5,zero_division=0),"roc_auc":roc_auc_score(yte,p)})
    comp=pd.DataFrame(rows); comp.to_csv(OUT/"model_comparison.csv",index=False)
    # Select using training-only CV AUC, never test AUC.
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    best_name=max(candidates, key=lambda n: np.mean([
        # cross-validated fit on training partition
        roc_auc_score(ytr.iloc[val], Pipeline([("prep",pre),("model",candidates[n])]).fit(Xtr.iloc[tr],ytr.iloc[tr]).predict_proba(Xtr.iloc[val])[:,1])
        for tr,val in cv.split(Xtr,ytr)
    ]))
    final=fitted[best_name]
    # Threshold chosen from out-of-fold training probabilities.
    from sklearn.model_selection import cross_val_predict
    oof=cross_val_predict(Pipeline([("prep",pre),("model",candidates[best_name])]),Xtr,ytr,cv=cv,method="predict_proba")[:,1]
    thresholds=np.arange(.20,.81,.01)
    threshold=max(thresholds,key=lambda t: recall_score(ytr,oof>=t,zero_division=0) - .25*max(0,.70-precision_score(ytr,oof>=t,zero_division=0)))
    # Single final test evaluation at the selected threshold.
    prob=final.predict_proba(Xte)[:,1]
    metrics={"selected_model":best_name,"threshold":float(threshold),"test_accuracy":float(accuracy_score(yte,prob>=threshold)),
             "test_precision":float(precision_score(yte,prob>=threshold,zero_division=0)),
             "test_recall":float(recall_score(yte,prob>=threshold,zero_division=0)),
             "test_roc_auc":float(roc_auc_score(yte,prob))}
    (OUT/"test_risk_scores.csv")
    id_col = x.iloc[split:]["appointment_id"].reset_index(drop=True) if "appointment_id" in x.columns else pd.Series(range(len(yte)), name="row_index")
    scores=pd.DataFrame({"appointment_id":id_col, "actual_no_show":yte.values,
                         "risk_score":prob,"predicted_no_show":(prob>=threshold).astype(int)})
    scores.to_csv(OUT/"test_risk_scores.csv",index=False)
    (OUT/"model_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(comp.to_string(index=False)); print(metrics)
    return final, metrics

ALLOWED={"illness_or_emergency","work_or_schedule","forgot_or_no_reminder","distance_or_transport","cost_or_fee","other"}
class FollowupInsight(BaseModel):
    model_config=ConfigDict(extra="forbid")
    root_cause: Literal["illness_or_emergency","work_or_schedule","forgot_or_no_reminder",
                        "distance_or_transport","cost_or_fee","other"]
    confidence: float = Field(ge=0,le=1)
    summary: str = Field(min_length=3)
    recommended_action: str = Field(min_length=3)

def analyze_note(note: str, client, model_name="gemini-2.0-flash", prompt_version="v2"):
    """Single-note Gemini call. API key is read from environment, never stored."""
    if prompt_version=="v1":
        prompt=("Classify this missed-appointment note. Return JSON only with root_cause, confidence, "
                "summary, recommended_action. Allowed root_cause values: "+", ".join(sorted(ALLOWED))+".\nNOTE: "+note)
    else:
        prompt=("You are a careful clinic follow-up analyst. Use only the note, do not infer demographics. "
                "Return one JSON object with exactly root_cause, confidence, summary, recommended_action. "
                "root_cause must be one of "+", ".join(sorted(ALLOWED))+". confidence is a number 0..1. "
                "Use a concise factual summary and a practical respectful action. Do not include markdown.\nNOTE: "+note)
    response=client.models.generate_content(model=model_name, contents=prompt,
        config={"response_mime_type":"application/json","temperature":0})
    return response.text

def run_part4():
    from google import genai
    path=DATA/"followup_notes.csv"
    if not path.exists(): raise FileNotFoundError(f"Missing {path}")
    notes=pd.read_csv(path).head(10)
    if "note_id" not in notes or "note_text" not in notes: raise ValueError("followup_notes.csv must contain note_id and note_text columns.")
    client=genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    if not os.environ.get("GEMINI_API_KEY"): raise EnvironmentError("Set GEMINI_API_KEY in your local environment.")
    rows=[]
    for _,r in notes.iterrows():
        note="" if pd.isna(r.note_text) else str(r.note_text)
        item={"note_id":r.note_id,"appointment_id":r.get("appointment_id",""),"root_cause":"","confidence":None,
              "summary":"","recommended_action":"","validation_status":"","raw_response":"",
              "analysis_method":"gemini_api"}
        if not note.strip():
            item["validation_status"]="skipped_blank"; rows.append(item); continue
        try:
            raw=analyze_note(note,client,prompt_version="v2")
            item["raw_response"]=raw
            data=json.loads(raw)
            insight=FollowupInsight.model_validate(data)
            if insight.root_cause not in ALLOWED: raise ValueError(f"root_cause not allowed: {insight.root_cause}")
            item.update(insight.model_dump()); item["validation_status"]="valid"
        except Exception as e:
            item["validation_status"]="invalid"
            item["validation_error"]=str(e)
        rows.append(item); time.sleep(1.0)
    pd.DataFrame(rows).to_csv(OUT/"notes_analysed.csv",index=False)
    print(pd.DataFrame(rows)[["note_id","root_cause","confidence","validation_status"]].to_string(index=False))
    return pd.DataFrame(rows)

def merge_recommendation():
    ap= pd.read_csv(OUT/"appointments_clean.csv")
    notes=pd.read_csv(OUT/"notes_analysed.csv")
    if "appointment_id" not in notes: raise ValueError("appointment_id absent from notes_analysed.csv")
    merged=ap.merge(notes,on="appointment_id",how="left",suffixes=("","_note"))
    merged.to_csv(OUT/"merged_recommendation.csv",index=False)
    if {"root_cause","reminder_sent"}.issubset(merged.columns):
        pd.crosstab(merged.root_cause,merged.reminder_sent).to_csv(OUT/"root_cause_by_reminder.csv")
    return merged
