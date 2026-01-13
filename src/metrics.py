import pandas as pd
from typing import List, Dict
"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""


def calculate_recall_at_k(
    df: pd.DataFrame,
    k_values: List[int]
) -> Dict[int, float]:
    """
    Calculate the Recall@k for each value of k in `k values`.

    Parameters:
        df (pd.DataFrame):
            DataFrame containing the columns 'code' (true code) and 'codes' (predicted codes).
        topk_values (List[int]):
            List of k values to calculate the Recall.

    Returns:
        Dict[int, float]: Dictionary with k values as keys and accuracies as values.

    Example:
        Input DataFrame:
        | code    | codes                 |
        |---------|-----------------------|
        | A1234   | [A1234, B5678, C9101] |
        | X9876   | [X9876, A1234, B5678] |

        Output:
        {1: 0.5, 2: 1.0, 3: 1.0}  # For Recall@1, Recall@2, Recall@3
    """

    # Initialize a dictionary to track accuracies for each k
    recalls = {k: 0 for k in k_values}

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        true_code = row['code']
        predicted_codes = row['codes']

        # Ensure uniqueness of predicted codes
        seen = set()
        unique_candidates = [x for x in predicted_codes if not (x in seen or seen.add(x))]

        # Check if true code is within the top-k candidates
        for k in k_values:
            if true_code in unique_candidates[:k]:
                recalls[k] += 1

    # Normalize by the total number of rows
    total_rows = len(df)
    if total_rows == 0:
        return {k: 0.0 for k in k_values}

    for k in k_values:
        recalls[k] = recalls[k] / total_rows

    return recalls


def calculate_norm(df_gs, df_preds):
    #Function adapted from https://github.com/TeMU-BSC/medprocner_evaluation_library/blob/main/utils.py
    list_gs_per_doc = df_gs.groupby('filename', group_keys=False).apply(lambda x: x[['filename', 'span_ini', 'span_end', "term", "label", "code"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename', group_keys=False).apply(lambda x: x[['filename', 'span_ini', 'span_end', "term", "label", "code"]].values.tolist()).to_list()
    scores = calculate_fscore(list_gs_per_doc, list_preds_per_doc, 'norm')
    return scores

def calculate_ner(df_gs, df_preds):
    #Function adapted from https://github.com/TeMU-BSC/medprocner_evaluation_library/blob/main/utils.py
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[["filename", 'span_ini', 'span_end', "term",  "label"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(lambda x: x[["filename", 'span_ini', 'span_end', "term", "label"]].values.tolist()).to_list()
    scores = calculate_fscore(list_gs_per_doc, list_preds_per_doc, 'ner')
    return scores


def calculate_fscore(gold_standard, predictions, task):
    # Code from https://github.com/TeMU-BSC/medprocner_evaluation_library/blob/main/utils.py
    """
    Calculate micro-averaged precision, recall and f-score from two pandas dataframe
    Depending on the task, do some different pre-processing to the data
    """


    # Cumulative true positives, false positives, false negatives
    total_tp, total_fp, total_fn = 0, 0, 0
    # Dictionary to store files in gold and prediction data.
    gs_files = {}
    pred_files = {}
    for document in gold_standard:
        document_id = document[0][0]
        gs_files[document_id] = document
    for document in predictions:
        document_id = document[0][0]
        pred_files[document_id] = document

    # Dictionary to store scores
    scores = {}

    # Iterate through documents in the Gold Standard
    for document_id in gs_files.keys():
        doc_tp, doc_fp, doc_fn = 0, 0, 0
        gold_doc = gs_files[document_id]
        #  Check if there are predictions for the current document, default to empty document if false
        if document_id not in pred_files.keys():
            predicted_doc = []
        else:
            predicted_doc = pred_files[document_id]
        if task == 'index':  # Separate codes
            gold_doc = list(set(gold_doc[0][1].split('+')))
            predicted_doc = list(set(predicted_doc[0][1].split('+'))) if predicted_doc else []
        # Iterate through a copy of our gold mentions
        for gold_annotation in gold_doc[:]:
            # Iterate through predictions looking for a match
            for prediction in predicted_doc[:]:
                # Separate possible composite normalizations
                if task == 'norm':
                    separate_prediction = prediction[:-1] + [code.rstrip() for code in sorted(str(prediction[-1]).split('+'))]  # Need to sort
                    separate_gold_annotation = gold_annotation[:-1] + [code.rstrip() for code in str(gold_annotation[-1]).split('+')]
                    if set(separate_gold_annotation) == set(separate_prediction):
                        # Add a true positive
                        doc_tp += 1
                        # Remove elements from list to calculate later false positives and false negatives
                        predicted_doc.remove(prediction)
                        gold_doc.remove(gold_annotation)
                        break
                if set(gold_annotation) == set(prediction):
                    # Add a true positive
                    doc_tp += 1
                    # Remove elements from list to calculate later false positives and false negatives
                    predicted_doc.remove(prediction)
                    gold_doc.remove(gold_annotation)
                    break
        # Get the number of false positives and false negatives from the items remaining in our lists
        doc_fp += len(predicted_doc)
        doc_fn += len(gold_doc)
        # Calculate document score
        try:
            precision = doc_tp / (doc_tp + doc_fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = doc_tp / (doc_tp + doc_fn)
        except ZeroDivisionError:
            recall = 0
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
        # Add to dictionary
        scores[document_id] = {"recall": round(recall, 4), "precision": round(precision, 4), "f_score": round(f_score, 4)}
        # Update totals
        total_tp += doc_tp
        total_fn += doc_fn
        total_fp += doc_fp

    # Now let's calculate the micro-averaged score using the cumulative TP, FP, FN
    try:
        precision = total_tp / (total_tp + total_fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = total_tp / (total_tp + total_fn)
    except ZeroDivisionError:
        recall = 0
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    scores['total'] = {"recall": round(recall, 4), "precision": round(precision, 4), "f_score": round(f_score, 4)}

    return scores