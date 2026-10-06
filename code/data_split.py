import numpy as np
import pandas as pd

def repository_temporal_split(df: pd.DataFrame, train_ratio: float = 0.70) -> pd.Series:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    assignments = pd.Series("train", index=df.index, dtype="string")
    for _, group in df.groupby("repository_id"):
        ordered = group.sort_values(["merged_at", "pr_id"], kind="stable")
        if len(ordered) < 3:
            continue
        cut = max(1, min(len(ordered) - 1, int(np.floor(len(ordered) * train_ratio))))
        assignments.loc[ordered.index[cut:]] = "test"
    return assignments
