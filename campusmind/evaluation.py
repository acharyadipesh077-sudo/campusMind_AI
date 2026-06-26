from typing import Any

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import precision_recall_fscore_support


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def bleu(prediction: str, reference: str) -> float:
    return float(
        sentence_bleu(
            [_tokens(reference)],
            _tokens(prediction),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def meteor(prediction: str, reference: str) -> float:
    try:
        return float(meteor_score([_tokens(reference)], _tokens(prediction)))
    except LookupError:
        nltk.download("wordnet", quiet=True)
        return float(meteor_score([_tokens(reference)], _tokens(prediction)))


def token_prf(prediction: str, reference: str) -> dict[str, float]:
    pred = set(_tokens(prediction))
    ref = set(_tokens(reference))
    labels = sorted(pred | ref)
    if not labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    y_pred = [1 if label in pred else 0 for label in labels]
    y_true = [1 if label in ref else 0 for label in labels]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def context_consistency(prediction: str, context: str | None) -> float:
    if not context:
        return 0.0
    pred_tokens = set(_tokens(prediction))
    context_tokens = set(_tokens(context))
    if not pred_tokens:
        return 0.0
    return len(pred_tokens & context_tokens) / len(pred_tokens)


def evaluate_item(prediction: str, reference: str, context: str | None = None) -> dict[str, Any]:
    scores = token_prf(prediction, reference)
    scores.update(
        {
            "bleu": bleu(prediction, reference),
            "meteor": meteor(prediction, reference),
            "context_consistency": context_consistency(prediction, context),
        }
    )
    return scores

