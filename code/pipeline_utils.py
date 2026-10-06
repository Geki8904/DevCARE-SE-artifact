import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from data_split import repository_temporal_split

FEATURES=['c1_code_churn','c2_method_complexity','c3_static_findings','c4_author_unfamiliarity']
def load_dataset(path='results/training/devcare_outcome_dataset.csv'):
    df=pd.read_csv(path); required={'repository_id','pr_id','merged_at','outcome_proxy','is_right_censored','analyzable_file_count',*FEATURES}
    if missing:=required-set(df): raise ValueError(f'missing required columns: {sorted(missing)}')
    if df.pr_id.duplicated().any(): raise ValueError('duplicate pr_id')
    df['merged_at']=pd.to_datetime(df.merged_at,errors='raise',utc=True)
    if df.outcome_proxy.nunique()!=2: raise ValueError('target must contain two classes')
    if (df[FEATURES[:3]]<0).any().any(): raise ValueError('C1-C3 cannot be negative')
    if (~df.c4_author_unfamiliarity.between(0,1)).any(): raise ValueError('C4 must be within [0,1]')
    if len(df)<100: raise ValueError('eligible rows < 100')
    df['split']=repository_temporal_split(df)
    return df
def queue_metric(y,p,fraction):
    y=np.asarray(y,int); p=np.asarray(p,float); k=max(1,math.ceil(fraction*len(y))); selected=y[np.argsort(-p,kind='stable')[:k]]; prevalence=y.mean()
    return {'fraction':fraction,'k':k,'positives_at_k':int(selected.sum()),'precision_at_k':float(selected.mean()),'recall_at_k':float(selected.sum()/y.sum()) if y.sum() else 0.0,'lift_at_k':float(selected.mean()/prevalence) if prevalence else 0.0,'test_prevalence':float(prevalence)}
def safe_auc(y,p):
    return (float(roc_auc_score(y,p)),float(average_precision_score(y,p))) if len(set(y))==2 else (None,None)
def write_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
