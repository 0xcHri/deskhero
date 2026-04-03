"""
=============================================================
 Deskhero - Accelerare il supporto aziendale tramite il triage automatico intelligente dei ticket
 Realizzato da Christian Cacchiotti
=============================================================

Classificazione automatica di ticket aziendali in categorie
(Tecnico / Amministrazione / Commerciale) e priorità
(alta / media / bassa).

Architettura:
  ▸ CATEGORIA: TF-IDF + SVM lineare
  ▸ PRIORITÀ:  decomposizione binaria a due stadi:
    - Stadio 1: "è alta priorità?" (sì/no)
    - Stadio 2: sui non-alta, "è bassa priorità?" (sì/no)
    - Media = ciò che non è né alta né bassa
    Approccio ispirato a Frank & Hall (2001), adatto a
    classificazione ordinale dove la classe centrale è
    definita per esclusione.

Confronto: LinearSVC, Random Forest, Logistic Regression.
Requisiti: vedi requirements.txt
"""

import json, os, re, warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ──────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.20
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tickets.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

STOPWORDS_IT = {
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "e", "o", "ma", "se", "che", "chi", "cui", "né",
    "è", "sono", "ha", "ho", "mi", "ti", "si", "ci", "vi",
    "me", "te", "lui", "lei", "noi", "voi", "loro",
    "questo", "quello", "questa", "quella", "questi", "quelli",
    "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "come", "dove", "quando", "perché", "anche", "ancora",
    "più", "molto", "poco", "tutto", "ogni", "altro",
    "già", "poi", "dopo", "prima", "sempre", "mai",
    "essere", "avere", "fare", "dire", "andare", "potere",
    "dovere", "volere", "sapere", "vedere", "venire", "dare",
    "stato", "stata", "stati", "state", "fatto", "fatta",
}
NEGATION_TRIGGERS = {"non", "nessun", "nessuna", "nessuno", "senza"}


# ══════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════

def preprocess_text(text):
    """Preprocessing: lowercase, pulizia, negazione italiana estesa,
    stemming leggero per normalizzare plurali (fatture→fattur = fattura→fattur)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    tokens = text.split()
    merged, i = [], 0
    while i < len(tokens):
        if tokens[i] in NEGATION_TRIGGERS:
            j = i + 1
            while j < len(tokens) and tokens[j] in STOPWORDS_IT:
                j += 1
            if j < len(tokens):
                merged.append(f"non_{_stem_it(tokens[j])}")
                i = j + 1
            else:
                i += 1
        else:
            merged.append(_stem_it(tokens[i]))
            i += 1
    return " ".join(t for t in merged if t not in STOPWORDS_IT and len(t) > 1)


def _stem_it(word):
    """Stemming italiano minimale: tronca la vocale finale per normalizzare
    singolare/plurale (fattura/fatture → fattur, errore/errori → error).
    Preserva parole corte e token composti (non_xxx)."""
    if word.startswith("non_"):
        return "non_" + _stem_it(word[4:])
    if len(word) <= 5:
        return word
    if word[-1] in "aeiou":
        return word[:-1]
    return word


# ══════════════════════════════════════════════════════════
#  FEATURE MANUALI DI URGENZA
#  Segnali paralinguistici che TF-IDF non pesa abbastanza:
#  keyword urgenza/calma, esclamativi, maiuscolo.
#  Vengono concatenate al vettore TF-IDF per dare al
#  classificatore un canale dedicato per l'urgenza.
# ══════════════════════════════════════════════════════════

URGENCY_WORDS = {"urgente", "urgentissimo", "bloccante", "critico", "emergenza",
                 "urgenza", "urge", "immediato", "gravissimo", "bloccato", "fermo",
                 "fermi", "panico", "disastro", "fretta", "subito"}
CALM_WORDS = {"calma", "comodo"}

def extract_urgency_features(texts_raw):
    """Estrae feature numeriche di urgenza dal testo raw (pre-preprocessing)."""
    features = []
    for text in texts_raw:
        features.append(_urgency_vector(text))
    return np.array(features, dtype=np.float32)

def _urgency_vector(text):
    """Calcola il vettore di urgenza per un singolo testo.
    Gestisce negazione: 'non urgente' non conta come urgenza."""
    # Pulizia punteggiatura per matching corretto
    clean = re.sub(r"[^\w\s]", " ", text.lower()).split()
    neg = {"non", "nessun", "nessuna", "nessuno", "senza"}

    n_urgency = 0
    n_calm = 0
    i = 0
    while i < len(clean):
        word = clean[i]
        # Se la parola precedente è una negazione, inverti il significato
        is_negated = (i > 0 and clean[i - 1] in neg)
        if word in URGENCY_WORDS:
            if is_negated:
                n_calm += 1   # "non urgente" → calma
            else:
                n_urgency += 1
        elif word in CALM_WORDS:
            if is_negated:
                n_urgency += 1  # "senza calma" → urgenza (raro ma corretto)
            else:
                n_calm += 1
        i += 1

    n_excl = text.count("!")
    ratio_upper = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return [n_urgency * 10.0, n_calm * 10.0, min(n_excl, 5) * 1.0, ratio_upper * 5.0]


# ══════════════════════════════════════════════════════════
#  ADDESTRAMENTO E CONFRONTO
# ══════════════════════════════════════════════════════════

def train_and_compare(X_train, X_test, y_train, y_test, task_name):
    """Addestra e confronta LinearSVC, Random Forest, Logistic Regression."""
    print(f"\n{'=' * 60}")
    print(f"  ADDESTRAMENTO — {task_name}")
    print(f"{'=' * 60}")
    results = {}

    # LinearSVC + GridSearchCV
    print(f"\n  ── LinearSVC + GridSearchCV ──")
    svm = LinearSVC(max_iter=10000, random_state=RANDOM_STATE,
                    class_weight="balanced", dual="auto")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(svm, {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
                        cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    results["SVM (LinearSVC)"] = {
        "model": best_svm, "y_pred": y_pred,
        "accuracy": acc, "f1_macro": f1, "best_params": grid.best_params_,
    }
    print(f"    Miglior C: {grid.best_params_['C']}  |  Acc: {acc:.4f}  |  F1: {f1:.4f}")

    # Random Forest
    print(f"\n  ── Random Forest ──")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="macro")
    results["Random Forest"] = {"model": rf, "y_pred": y_pred_rf,
                                "accuracy": acc_rf, "f1_macro": f1_rf}
    print(f"    Acc: {acc_rf:.4f}  |  F1: {f1_rf:.4f}")

    # Logistic Regression
    print(f"\n  ── Logistic Regression ──")
    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE,
                            class_weight="balanced", solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average="macro")
    results["Logistic Regression"] = {"model": lr, "y_pred": y_pred_lr,
                                      "accuracy": acc_lr, "f1_macro": f1_lr}
    print(f"    Acc: {acc_lr:.4f}  |  F1: {f1_lr:.4f}")

    print(f"\n  {'Modello':<25} {'Accuracy':>10} {'F1 macro':>10}")
    print(f"  {'─' * 47}")
    for name, res in sorted(results.items(), key=lambda x: -x[1]["f1_macro"]):
        print(f"  {name:<25} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f}")
    return results, y_test


def build_cascade_features(svm_cat, X_train, X_test, y_train_cat, cat_labels):
    """Cascata: one-hot della categoria predetta concatenata a TF-IDF."""
    print(f"\n  Costruzione feature cascata...")
    cat_pred_train = cross_val_predict(svm_cat, X_train, y_train_cat, cv=5, method="predict")
    cat_pred_test = svm_cat.predict(X_test)
    def one_hot(preds, labels):
        oh = np.zeros((len(preds), len(labels)), dtype=np.float32)
        idx_map = {l: i for i, l in enumerate(labels)}
        for i, p in enumerate(preds):
            oh[i, idx_map[p]] = 1.0
        return csr_matrix(oh)
    X_train_aug = hstack([X_train, one_hot(cat_pred_train, cat_labels)])
    X_test_aug = hstack([X_test, one_hot(cat_pred_test, cat_labels)])
    print(f"  Feature: {X_train_aug.shape[1]} (TF-IDF + {len(cat_labels)} one-hot)")
    return X_train_aug, X_test_aug


def train_binary_svm(X_train, X_test, y_train, y_test, label):
    """Addestra un singolo SVM binario con GridSearchCV."""
    svm = LinearSVC(max_iter=10000, random_state=RANDOM_STATE,
                    class_weight="balanced", dual="auto")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(svm, {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
                        cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"    {label}: C={grid.best_params_['C']}  Acc={acc:.4f}  F1={f1:.4f}")
    return model, y_pred


def predict_priority_two_stage(X, svm_alta, svm_bassa):
    """Predizione a due stadi: alta? → se no, bassa? → se no, media."""
    pred_alta = svm_alta.predict(X)
    pred_bassa = svm_bassa.predict(X)
    n = X.shape[0]
    result = []
    for i in range(n):
        if pred_alta[i] == 1:
            result.append("alta")
        elif pred_bassa[i] == 1:
            result.append("bassa")
        else:
            result.append("media")
    return np.array(result)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 60)
    print("  TICKET TRIAGE — Pipeline ML")
    print("  Categoria: TF-IDF + SVM")
    print("  Priorità:  Decomposizione binaria a due stadi")
    print("█" * 60)

    # Caricamento
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print(f"\n  Dataset: {len(df)} ticket")
    print(f"  Categorie: {dict(df['category'].value_counts())}")
    print(f"  Priorità:  {dict(df['priority'].value_counts())}")

    # Preprocessing (title weighting 2×)
    df["text_clean"] = (
        df["title"].astype(str) + " " + df["title"].astype(str) + " " + df["body"].astype(str)
    ).apply(preprocess_text)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, dtype=np.float32)
    X_tfidf = vectorizer.fit_transform(df["text_clean"])
    print(f"  TF-IDF: {X_tfidf.shape[1]} feature")

    # Feature manuali di urgenza (dal testo raw, pre-preprocessing)
    df["text_raw"] = df["title"].astype(str) + " " + df["body"].astype(str)
    X_urgency = extract_urgency_features(df["text_raw"].tolist())
    print(f"  Feature urgenza: {X_urgency.shape[1]} (urgency, calm, excl, upper)")

    # Split
    cat_labels = sorted(df["category"].unique().tolist())
    y_cat, y_pri = df["category"].values, df["priority"].values
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cat)
    X_train, X_test = X_tfidf[idx_train], X_tfidf[idx_test]
    y_train_cat, y_test_cat = y_cat[idx_train], y_cat[idx_test]
    y_train_pri, y_test_pri = y_pri[idx_train], y_pri[idx_test]
    print(f"  Split: {len(idx_train)} train / {len(idx_test)} test")

    # ── TASK A: CATEGORIA ──
    print("\n" + "█" * 60)
    print("  TASK A — CATEGORIA (TF-IDF + SVM)")
    print("█" * 60)
    results_cat, y_test_cat = train_and_compare(
        X_train, X_test, y_train_cat, y_test_cat, "Categoria")
    for name, res in results_cat.items():
        print(f"\n  ── {name} ──")
        print(classification_report(y_test_cat, res["y_pred"], digits=4, zero_division=0))

    # ── TASK B: PRIORITÀ — Decomposizione binaria a due stadi ──
    #   Confronto tra SVM, Random Forest e Logistic Regression
    #   su entrambi gli stadi della decomposizione
    print("\n" + "█" * 60)
    print("  TASK B — PRIORITÀ (Decomposizione binaria a due stadi)")
    print("  Stadio 1: alta vs non-alta")
    print("  Stadio 2: bassa vs media (solo sui non-alta)")
    print("█" * 60)

    svm_cat = results_cat["SVM (LinearSVC)"]["model"]
    X_aug_train, X_aug_test = build_cascade_features(
        svm_cat, X_train, X_test, y_train_cat, cat_labels)

    # Concatena feature manuali di urgenza al vettore cascata
    X_urg_train = csr_matrix(X_urgency[idx_train])
    X_urg_test = csr_matrix(X_urgency[idx_test])
    X_aug_train = hstack([X_aug_train, X_urg_train])
    X_aug_test = hstack([X_aug_test, X_urg_test])
    print(f"  Feature totali priorità: {X_aug_train.shape[1]} (TF-IDF + cascata + urgenza)")

    # Label binarie
    y_train_is_alta = (y_train_pri == "alta").astype(int)
    y_test_is_alta = (y_test_pri == "alta").astype(int)
    mask_train_non_alta = y_train_pri != "alta"
    mask_test_non_alta = y_test_pri != "alta"
    y_train_is_bassa = (y_train_pri[mask_train_non_alta] == "bassa").astype(int)
    y_test_is_bassa = (y_test_pri[mask_test_non_alta] == "bassa").astype(int)

    print(f"\n  Stadio 1 — alta ({y_train_is_alta.sum()}) vs non-alta ({(~y_train_is_alta.astype(bool)).sum()})")
    print(f"  Stadio 2 — bassa ({y_train_is_bassa.sum()}) vs media ({(~y_train_is_bassa.astype(bool)).sum()})")

    # Definizione classificatori
    classifiers = {
        "SVM (LinearSVC)": lambda: LinearSVC(max_iter=10000, random_state=RANDOM_STATE,
                                              class_weight="balanced", dual="auto"),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                         random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression": lambda: LogisticRegression(max_iter=5000, random_state=RANDOM_STATE,
                                                           class_weight="balanced", solver="lbfgs"),
    }

    results_2stage = {}
    best_models = {}

    for clf_name, clf_factory in classifiers.items():
        print(f"\n  ── {clf_name} ──")

        # Stadio 1: alta vs non-alta
        model_alta = clf_factory()
        if clf_name == "SVM (LinearSVC)":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            grid = GridSearchCV(model_alta, {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
                                cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
            grid.fit(X_aug_train, y_train_is_alta)
            model_alta = grid.best_estimator_
            print(f"    Stadio 1 (alta): C={grid.best_params_['C']}")
        else:
            model_alta.fit(X_aug_train, y_train_is_alta)

        # Stadio 2: bassa vs media
        model_bassa = clf_factory()
        if clf_name == "SVM (LinearSVC)":
            grid2 = GridSearchCV(model_bassa, {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
                                 cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
            grid2.fit(X_aug_train[mask_train_non_alta], y_train_is_bassa)
            model_bassa = grid2.best_estimator_
            print(f"    Stadio 2 (bassa): C={grid2.best_params_['C']}")
        else:
            model_bassa.fit(X_aug_train[mask_train_non_alta], y_train_is_bassa)

        # Predizione combinata
        y_pred = predict_priority_two_stage(X_aug_test, model_alta, model_bassa)
        acc = accuracy_score(y_test_pri, y_pred)
        f1 = f1_score(y_test_pri, y_pred, average="macro")
        results_2stage[clf_name] = {"accuracy": acc, "f1_macro": f1, "y_pred": y_pred}
        best_models[clf_name] = (model_alta, model_bassa)
        print(f"    Risultato: Acc={acc:.4f}  |  F1={f1:.4f}")

    # Tabella confronto
    print(f"\n  {'═' * 55}")
    print(f"  CONFRONTO PRIORITÀ — Due stadi")
    print(f"  {'═' * 55}")
    print(f"  {'Classificatore':<25} {'Accuracy':>10} {'F1 macro':>10}")
    print(f"  {'─' * 47}")
    best_clf = max(results_2stage, key=lambda k: results_2stage[k]["f1_macro"])
    for name, res in sorted(results_2stage.items(), key=lambda x: -x[1]["f1_macro"]):
        marker = " ★" if name == best_clf else ""
        print(f"  {name:<25} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f}{marker}")

    # Classification report del migliore
    print(f"\n  ── Classification Report ({best_clf}) ──")
    print(classification_report(y_test_pri, results_2stage[best_clf]["y_pred"],
                                digits=4, zero_division=0))

    # ── SALVATAGGIO (SVM, il più interpretabile per la dashboard) ──
    print(f"\n{'=' * 60}")
    print(f"  SALVATAGGIO MODELLI → {MODELS_DIR}")
    print(f"{'=' * 60}")
    svm_cat_final = results_cat["SVM (LinearSVC)"]["model"]
    svm_alta, svm_bassa = best_models["SVM (LinearSVC)"]

    joblib.dump(svm_cat_final, os.path.join(MODELS_DIR, "svm_category.joblib"))
    joblib.dump(svm_alta, os.path.join(MODELS_DIR, "svm_is_alta.joblib"))
    joblib.dump(svm_bassa, os.path.join(MODELS_DIR, "svm_is_bassa.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    print(f"  ✓ svm_category.joblib")
    print(f"  ✓ svm_is_alta.joblib  (stadio 1)")
    print(f"  ✓ svm_is_bassa.joblib (stadio 2)")
    print(f"  ✓ tfidf_vectorizer.joblib")

    metadata = {
        "architecture": {
            "category": "TF-IDF + LinearSVC",
            "priority": "Decomposizione binaria a due stadi (alta→bassa→media)",
        },
        "category_labels": cat_labels,
        "priority_labels": ["alta", "bassa", "media"],
        "results": {
            "category": {n: {"accuracy": r["accuracy"], "f1": r["f1_macro"]}
                         for n, r in results_cat.items()},
            "priority_two_stage": {n: {"accuracy": r["accuracy"], "f1": r["f1_macro"]}
                                   for n, r in results_2stage.items()},
        },
        "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ metadata.json")

    print(f"\n{'█' * 60}")
    print(f"  PIPELINE COMPLETATA")
    print(f"{'█' * 60}\n")


if __name__ == "__main__":
    main()
