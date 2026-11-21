import json, pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.schema import schema

df = schema.validate(pd.read_csv("data/heart_failure_clinical_records_dataset.csv"))
train, test = train_test_split(df, test_size=0.25, random_state=42,
                               stratify=df["DEATH_EVENT"])

# Cox model
cox = CoxPHFitter(penalizer=0.1).fit(train, "time", "DEATH_EVENT")
cox_c = concordance_index(test["time"], -cox.predict_partial_hazard(test),
                          test["DEATH_EVENT"])

# Plain classifier baseline (ignores time/censoring)
X = df.drop(columns=["time", "DEATH_EVENT"])
clf = LogisticRegression(max_iter=1000).fit(X.loc[train.index], train["DEATH_EVENT"])
risk = clf.predict_proba(X.loc[test.index])[:, 1]
clf_c = concordance_index(test["time"], -risk, test["DEATH_EVENT"])
clf_auc = roc_auc_score(test["DEATH_EVENT"], risk)

metrics = {"cox_cindex": round(cox_c, 3),
           "classifier_cindex": round(clf_c, 3),
           "classifier_auc": round(clf_auc, 3)}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(metrics)

# Quality gate: fail the workflow if the model degrades
assert cox_c >= 0.70, f"Cox C-index dropped to {cox_c:.3f}"