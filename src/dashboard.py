"""
=============================================================
 Deskhero - Accelerare il supporto aziendale tramite il triage automatico intelligente dei ticket
 Realizzato da Christian Cacchiotti
=============================================================

  Interfaccia Gradio per la classificazione automatica di ticket
  in categoria (Tecnico / Amministrazione / Commerciale) e
  priorità (alta / media / bassa).

  Architettura:
    - Categoria: TF-IDF + SVM lineare
    - Priorità:  Decomposizione binaria a due stadi + feature urgenza

  Tab:
    [1] Classificazione singolo ticket + top-5 parole + storico
    [2] Classificazione batch da CSV con export

  Requisiti: vedi requirements.txt
"""

import os, json, re, tempfile, warnings
from collections import Counter

import gradio as gr
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import joblib
    from scipy.sparse import hstack, csr_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠  scikit-learn / joblib / scipy non trovati.")


# ─────────────────────────────────────────────────────────
#  CONFIGURAZIONE
# ─────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "..", "models")
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "tickets.csv")

CATEGORY_LABELS = ["Amministrazione", "Commerciale", "Tecnico"]
PRIORITY_LABELS = ["alta", "media", "bassa"]

# Design system — colori coerenti
CAT_COLORS = {"Amministrazione": "#EC539F", "Commerciale": "#8B5CF6", "Tecnico": "#3B82F6"}
PRI_COLORS = {"alta": "#EF4444", "media": "#F59E0B", "bassa": "#10B981"}
CAT_ICONS  = {"Amministrazione": "🗃️", "Commerciale": "💼", "Tecnico": "🔧"}
PRI_ICONS  = {"alta": "🔴", "media": "🟡", "bassa": "🟢"}

STOPWORDS_IT = {
    "di","a","da","in","con","su","per","tra","fra","il","lo","la","i","gli","le",
    "un","uno","una","e","o","ma","se","che","chi","cui","né","è","sono","ha","ho",
    "mi","ti","si","ci","vi","me","te","lui","lei","noi","voi","loro","questo","quello",
    "questa","quella","questi","quelli","del","dello","della","dei","degli","delle",
    "al","allo","alla","ai","agli","alle","dal","dallo","dalla","dai","dagli","dalle",
    "nel","nello","nella","nei","negli","nelle","sul","sullo","sulla","sui","sugli","sulle",
    "come","dove","quando","perché","anche","ancora","più","molto","poco","tutto","ogni",
    "altro","già","poi","dopo","prima","sempre","mai","essere","avere","fare","dire",
    "andare","potere","dovere","volere","sapere","vedere","venire","dare",
    "stato","stata","stati","state","fatto","fatta",
}

URGENCY_WORDS = {"urgente","urgentissimo","bloccante","critico","emergenza",
                 "urgenza","urge","immediato","gravissimo","bloccato","fermo",
                 "fermi","panico","disastro","fretta","subito"}
CALM_WORDS = {"calma","comodo"}


# ─────────────────────────────────────────────────────────
#  FEATURE URGENZA + NEGAZIONI + STEMMING + PREPROCESSING
# ─────────────────────────────────────────────────────────

def extract_urgency_single(text_raw):
    clean = re.sub(r"[^\w\s]", " ", text_raw.lower()).split()
    neg = {"non","nessun","nessuna","nessuno","senza"}
    nu = nc = 0
    for i, w in enumerate(clean):
        negated = (i > 0 and clean[i-1] in neg)
        if w in URGENCY_WORDS:
            if negated: nc += 1
            else: nu += 1
        elif w in CALM_WORDS:
            if negated: nu += 1
            else: nc += 1
    ne = text_raw.count("!")
    ru = sum(1 for c in text_raw if c.isupper()) / max(len(text_raw), 1)
    return np.array([[nu*10.0, nc*10.0, min(ne,5)*1.0, ru*5.0]], dtype=np.float32)

def _stem_it(word):
    """Stemming italiano minimale: tronca la vocale finale per normalizzare
    singolare/plurale (fattura/fatture → fattur, errore/errori → error)."""
    if word.startswith("non_"):
        return "non_" + _stem_it(word[4:])
    if len(word) <= 5:
        return word
    if word[-1] in "aeiou":
        return word[:-1]
    return word

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    tokens = text.split()
    merged, i = [], 0
    while i < len(tokens):
        if tokens[i] in {"non","nessun","nessuna","nessuno","senza"}:
            j = i + 1
            while j < len(tokens) and tokens[j] in STOPWORDS_IT:
                j += 1
            if j < len(tokens):
                merged.append(f"non_{_stem_it(tokens[j])}"); i = j + 1
            else: i += 1
        else:
            merged.append(_stem_it(tokens[i])); i += 1
    return " ".join(t for t in merged if t not in STOPWORDS_IT and len(t) > 1)


# ─────────────────────────────────────────────────────────
#  CARICAMENTO MODELLI
# ─────────────────────────────────────────────────────────

def _find(fn):
    for d in [MODELS_DIR, BASE_DIR]:
        p = os.path.join(d, fn)
        if os.path.isfile(p): return p
    return None

def load_artifacts():
    A = {"cat": None, "alta": None, "bassa": None, "vec": None, "meta": None}
    if not SKLEARN_AVAILABLE: return A
    for k, fn in [("cat","svm_category.joblib"),("alta","svm_is_alta.joblib"),
                   ("bassa","svm_is_bassa.joblib"),("vec","tfidf_vectorizer.joblib")]:
        p = _find(fn)
        if p: A[k] = joblib.load(p); print(f"✓ {k}")
        else: print(f"✗ {fn} non trovato")
    p = _find("metadata.json")
    if p:
        with open(p, "r", encoding="utf-8") as f: A["meta"] = json.load(f)
    return A

ART = load_artifacts()


# ─────────────────────────────────────────────────────────
#  LOGICA DI CLASSIFICAZIONE
# ─────────────────────────────────────────────────────────

def _top_words(model, X, pred_class, fnames, n=5):
    raw = model
    if hasattr(model, "calibrated_classifiers_"):
        raw = model.calibrated_classifiers_[0].estimator
    if not hasattr(raw, "coef_"):
        tfidf = X.toarray().flatten()
        top_idx = np.argsort(tfidf)[::-1][:n]
        return [(fnames[i], round(float(tfidf[i]),4)) for i in top_idx if tfidf[i]>0]
    tfidf = X.toarray().flatten()
    classes = list(raw.classes_)
    cidx = classes.index(pred_class)
    coefs = raw.coef_[cidx]
    if hasattr(coefs, "toarray"): coefs = coefs.toarray().flatten()
    else: coefs = np.asarray(coefs).flatten()
    coefs = coefs[:len(tfidf)]
    nz = tfidf.nonzero()[0]
    infl = [(fnames[i], coefs[i]*tfidf[i]) for i in nz]
    infl.sort(key=lambda x: x[1], reverse=True)
    return [(w, round(s,4)) for w,s in infl[:n]]


def _classify(title, body):
    """Core classification: returns (category, priority, top_words) or raises."""
    vec, cat_m, alta_m, bassa_m = ART["vec"], ART["cat"], ART["alta"], ART["bassa"]
    meta = ART["meta"]
    raw = f"{title} {title} {body}".strip()
    X = vec.transform([preprocess_text(raw)])
    fnames = vec.get_feature_names_out()
    cat = cat_m.predict(X)[0]
    cl = meta.get("category_labels", CATEGORY_LABELS) if meta else CATEGORY_LABELS
    oh = np.zeros((1, len(cl)), dtype=np.float32)
    if cat in cl: oh[0, cl.index(cat)] = 1.0
    Xurg = csr_matrix(extract_urgency_single(f"{title} {body}".strip()))
    Xa = hstack([X, csr_matrix(oh), Xurg])
    if alta_m.predict(Xa)[0] == 1: pri = "alta"
    elif bassa_m.predict(Xa)[0] == 1: pri = "bassa"
    else: pri = "media"
    tw = _top_words(cat_m, X, cat, fnames, n=5)
    return cat, pri, tw


SESSION_HISTORY = []


# ─────────────────────────────────────────────────────────
#  HTML BUILDERS — Design system
# ─────────────────────────────────────────────────────────

def _card(label, value, sublabel, color, icon):
    return f"""
    <div style="flex:1; min-width:220px; background:linear-gradient(135deg, {color}12, {color}06);
                border:1px solid {color}30; border-radius:16px; padding:28px 24px;
                text-align:center; position:relative; overflow:hidden;
                box-shadow: 0 4px 24px {color}15;
                transition: transform 0.2s ease, box-shadow 0.2s ease;">
      <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
      <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase;
                  letter-spacing:0.15em; font-weight:600; margin-bottom:6px;">{label}</div>
      <div style="font-size:1.6rem; font-weight:800; color:{color};
                  letter-spacing:-0.02em;">{value}</div>
      <div style="font-size:0.65rem; color:#94A3B8; margin-top:6px;
                  font-weight:500;">{sublabel}</div>
    </div>"""


def _word_chips(words, color):
    if not words:
        return '<span style="color:#64748B; font-size:0.82rem; font-style:italic;">Nessuna parola disponibile</span>'
    mx = max(abs(s) for _,s in words) or 1
    chips = ""
    for w, s in words:
        pct = min(abs(s)/mx, 1.0)
        opacity = int(pct * 35 + 10)
        chips += (f'<span style="display:inline-block; padding:6px 14px; margin:3px; '
                  f'border-radius:20px; background:{color}{opacity:02x}; '
                  f'border:1px solid {color}25; font-size:0.82rem; font-weight:600; '
                  f'color:{color}; letter-spacing:0.01em;">'
                  f'{w} <span style="font-weight:400; opacity:0.7;">{s:.4f}</span></span>')
    return chips


def _history_table():
    if not SESSION_HISTORY:
        return ('<div style="text-align:center; padding:40px 20px; color:#64748B; '
                'font-size:0.85rem;">Nessun ticket classificato in questa sessione</div>')
    rows = ""
    for t in reversed(SESSION_HISTORY[-10:]):
        cc = CAT_COLORS.get(t["cat"], "#64748B")
        pc = PRI_COLORS.get(t["pri"], "#64748B")
        ci = CAT_ICONS.get(t["cat"], "")
        pi = PRI_ICONS.get(t["pri"], "")
        rows += f"""<tr style="border-bottom:1px solid #E2E8F0; transition:background 0.15s;">
          <td style="padding:10px 12px; font-size:0.82rem;
                     max-width:200px; overflow:hidden; text-overflow:ellipsis;
                     white-space:nowrap; font-weight:500;">{t['title']}</td>
          <td style="padding:10px 12px; font-size:0.8rem;
                     max-width:280px; overflow:hidden; text-overflow:ellipsis;
                     white-space:nowrap;">{t['body']}</td>
          <td style="padding:10px 12px; text-align:center;">
            <span style="color:{cc}; font-weight:700; font-size:0.82rem;">{ci} {t['cat']}</span></td>
          <td style="padding:10px 12px; text-align:center;">
            <span style="background:{pc}15; color:{pc}; padding:4px 12px; border-radius:12px;
                         font-weight:700; font-size:0.78rem; text-transform:uppercase;
                         letter-spacing:0.05em;">{t['pri']}</span></td>
        </tr>"""
    return f"""
    <table style="width:100%; border-collapse:collapse;">
      <thead><tr style="border-bottom:2px solid #CBD5E1;">
        <th style="padding:10px 12px; text-align:left; color:#64748B; font-weight:700;
                   font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">Titolo</th>
        <th style="padding:10px 12px; text-align:left; color:#64748B; font-weight:700;
                   font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">Descrizione</th>
        <th style="padding:10px 12px; text-align:center; color:#64748B; font-weight:700;
                   font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">Categoria</th>
        <th style="padding:10px 12px; text-align:center; color:#64748B; font-weight:700;
                   font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">Priorità</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div style="text-align:center; margin-top:10px; color:#94A3B8; font-size:0.72rem; font-weight:500;">
      {len(SESSION_HISTORY)} ticket classificati in questa sessione
    </div>"""


# ─────────────────────────────────────────────────────────
#  HANDLER TAB 1 — Classifica ticket singolo
# ─────────────────────────────────────────────────────────

_EMPTY = '<div style="min-height:100px;"></div>'

def classify_single(title, body):
    if not all([ART["vec"], ART["cat"], ART["alta"], ART["bassa"]]):
        return (f'<div style="min-height:100px; padding:30px; text-align:center; color:#EF4444; '
                f'font-weight:600;">⚠️ Modelli non caricati</div>', "", "", _history_table())
    if not title.strip() and not body.strip():
        return (f'<div style="min-height:100px; padding:30px; text-align:center; color:#F59E0B; '
                f'font-weight:500;">Inserisci almeno il titolo o la descrizione del ticket.</div>',
                "", "", _history_table())

    cat, pri, tw = _classify(title, body)
    cc = CAT_COLORS.get(cat, "#64748B")
    pc = PRI_COLORS.get(pri, "#64748B")
    ci = CAT_ICONS.get(cat, "")
    pi = PRI_ICONS.get(pri, "")

    result = f"""<div style="display:flex; gap:24px; justify-content:center; flex-wrap:wrap;
                             margin:8px 0; animation: fadeIn 0.4s ease;">
        {_card("Categoria", cat, "TF-IDF + SVM", cc, ci)}
        {_card("Priorità", pri.upper(), "CLASSIFICAZIONE BINARIA + SVM", pc, pi)}
    </div>"""

    words_html = _word_chips(tw, cc)

    SESSION_HISTORY.append({
        "title": title[:60] + ("…" if len(title) > 60 else ""),
        "body": body[:80] + ("…" if len(body) > 80 else ""),
        "cat": cat, "pri": pri,
    })

    return result, words_html, "", _history_table()


# ─────────────────────────────────────────────────────────
#  HANDLER TAB 2 — Batch CSV
# ─────────────────────────────────────────────────────────

def classify_batch(file):
    if not all([ART["vec"], ART["cat"], ART["alta"], ART["bassa"]]):
        return "⚠️ Modelli non caricati.", None
    if file is None:
        return "⚠️ Nessun file caricato.", None

    fp = file if isinstance(file, str) else getattr(file, "name", file)
    try: df = pd.read_csv(fp, encoding="utf-8")
    except Exception:
        try: df = pd.read_csv(fp, encoding="latin-1")
        except Exception as e: return f"⚠️ Errore: {e}", None

    df.columns = df.columns.str.strip().str.lower()
    tcol = next((c for c in df.columns if c in ["title","titolo","oggetto"]), None)
    bcol = next((c for c in df.columns if c in ["body","descrizione","description","testo"]), None)
    if not tcol and not bcol:
        return "⚠️ Colonne non trovate. Servono <code>title</code> e/o <code>body</code>.", None

    vec, cat_m, alta_m, bassa_m = ART["vec"], ART["cat"], ART["alta"], ART["bassa"]
    meta = ART["meta"]
    cl = meta.get("category_labels", CATEGORY_LABELS) if meta else CATEGORY_LABELS

    clean, raws = [], []
    for _, r in df.iterrows():
        t = str(r.get(tcol,"")) if tcol else ""
        b = str(r.get(bcol,"")) if bcol else ""
        raws.append(f"{t} {b}".strip())
        clean.append(preprocess_text(f"{t} {t} {b}".strip()))

    tfidf = vec.transform(clean)
    cats = cat_m.predict(tfidf)
    df["predicted_category"] = cats

    oh = np.zeros((len(cats), len(cl)), dtype=np.float32)
    for i, c in enumerate(cats):
        if c in cl: oh[i, cl.index(c)] = 1.0
    urg = np.array([extract_urgency_single(t)[0] for t in raws], dtype=np.float32)
    Xa = hstack([tfidf, csr_matrix(oh), csr_matrix(urg)])

    pa, pb = alta_m.predict(Xa), bassa_m.predict(Xa)
    pris = ["alta" if a==1 else "bassa" if b==1 else "media" for a,b in zip(pa,pb)]
    df["predicted_priority"] = pris

    out = os.path.join(tempfile.gettempdir(), "predizioni_batch.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # Anteprima HTML
    cols = []
    if "id" in df.columns: cols.append("id")
    if tcol: cols.append(tcol)
    cols += ["predicted_category", "predicted_priority"]
    preview = df[cols].head(15)

    rows = ""
    for _, r in preview.iterrows():
        cat_v = r.get("predicted_category","")
        pri_v = r.get("predicted_priority","")
        cc = CAT_COLORS.get(cat_v, "#64748B")
        pc = PRI_COLORS.get(pri_v, "#64748B")
        cells = ""
        for c in cols:
            v = r[c]
            if c == "predicted_category":
                cells += f'<td style="padding:8px 12px;"><span style="color:{cc}; font-weight:700;">{v}</span></td>'
            elif c == "predicted_priority":
                cells += (f'<td style="padding:8px 12px;"><span style="background:{pc}15; color:{pc}; '
                          f'padding:3px 10px; border-radius:10px; font-weight:700; font-size:0.78rem; '
                          f'text-transform:uppercase;">{v}</span></td>')
            else:
                cells += f'<td style="padding:8px 12px; font-size:0.82rem; max-width:250px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{v}</td>'
        rows += f'<tr style="border-bottom:1px solid #E2E8F0;">{cells}</tr>'

    hdr = "".join(f'<th style="padding:10px 12px; text-align:left; color:#64748B; font-weight:700; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">{c}</th>' for c in cols)

    html = f"""
    <div style="width:100%; margin-top:16px; animation: fadeIn 0.4s ease;">
      <div style="text-align:center; margin-bottom:16px;">
        <span style="background:linear-gradient(135deg, #1D4ED8, #3B82F6); color:white;
                     padding:8px 20px; border-radius:20px; font-weight:700; font-size:0.85rem;
                     letter-spacing:0.02em;">
          ✓ Classificati {len(df)} ticket
        </span>
      </div>
      <table style="width:100%; border-collapse:collapse; border-radius:12px; overflow:hidden;
                    box-shadow: 0 1px 8px rgba(0,0,0,0.10);">
        <thead><tr style="background:#F1F5F9; border-bottom:2px solid #CBD5E1;">{hdr}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div style="text-align:center; margin-top:8px; color:#94A3B8; font-size:0.80rem;">
        Anteprima prime {min(15, len(df))} righe
      </div>
    </div>"""
    return html, out


# ─────────────────────────────────────────────────────────
#  INTERFACCIA GRADIO CON CUSTOM CSS
# ─────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Montserrat', sans-serif !important; }

.gradio-container {
    max-width: 1080px !important;
    width: 100% !important;
    min-width: 900px !important;
    margin: 0 auto !important;

}

/* Header area */
.app-header {
    background: linear-gradient(135deg, #1D4ED8 0%, #1D4ED8 50%, #1D4ED8 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
    align-items: center;
}
.app-header::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.15) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(139,92,246,0.1) 0%, transparent 50%);
    pointer-events: none;
}

/* Tab styling */
.tab-nav { border-bottom: 2px solid #E2E8F0 !important; margin-bottom: 4px !important; }
.tab-nav button {
    font-color: #3B82F6 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    padding: 12px 24px !important;
    border-radius: 12px 12px 0 0 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    color: white !important;
    border: none !important;
}

/* Inputs */
textarea, input[type="text"] {
    border-radius: 12px !important;
    border: 2px solid #E2E8F0 !important;
    transition: border-color 0.2s ease !important;
    font-size: 0.9rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* Primary button */
.primary {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    font-size: 0.95rem !important;
    padding: 12px 0 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}

/* Fixed layout */
.tabs, .tabitem, .tabitem > div { width: 100% !important; }
.tabitem { min-height: 500px !important; }
footer { display: none !important; }

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
"""


def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="Deskhero - Dashboard", theme=gr.themes.Soft(primary_hue="blue")) as app:
        # ── HEADER ──
        gr.HTML("""
        <div class="app-header">
          <div style="position:relative; z-index:1; text-align:center; justify-content: center; align-items: center;">
            <img src="https://i.imgur.com/FCd819H.png" width="150" style="display: block; margin: 0 auto;" alt="DeskHero Logo"><br>
            <h1 style="font-size:1.7rem; font-weight:750; color:white; margin:0;
                        letter-spacing:-0.03em;">
              Dashboard interattiva per la classificazione dei ticket
            </h1>
            <p style="font-size:0.85rem; color:#FFFFFF; margin-top:8px; font-weight:500;
                      letter-spacing:0.02em;">
              Inserisci un ticket per classificare automaticamente categoria e priorità
            </p>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: CLASSIFICA ──
            with gr.TabItem("Classifica Ticket", id="single"):
                gr.HTML('<div style="height:8px; font-color: #3B82F6;"></div>')

                with gr.Column():
                    input_title = gr.Textbox(
                        label="Titolo del ticket",
                        placeholder="Es: Errore nell'emissione della fattura mensile",
                        lines=1, max_lines=1,
                    )
                    input_body = gr.Textbox(
                        label="Descrizione",
                        placeholder="Es: Da tre giorni il sistema di fatturazione restituisce un errore 500 quando si tenta di generare fatture",
                        lines=5,
                    )
                    classify_btn = gr.Button("⚡ Classifica Ticket", variant="primary", size="lg")

                result_html = gr.HTML(value=_EMPTY)

                gr.HTML("""
                <div style="text-align:center; margin:16px 0 8px;">
                    <span style="font-size:0.8rem; font-weight:700; color:#475569;
                                 text-transform:uppercase; letter-spacing:0.12em;">
                      📌 Top 5 parole più influenti
                    </span>
                </div>""")
                words_html = gr.HTML(
                    value='<div style="text-align:center; padding:16px; color:#94A3B8; font-size:0.82rem;">Classifica un ticket per vedere le parole influenti</div>'
                )
                pri_words = gr.HTML(visible=False)

                gr.HTML("""
                <div style="text-align:center; margin:24px 0 8px;">
                    <span style="font-size:0.8rem; font-weight:700; color:#475569;
                                 text-transform:uppercase; letter-spacing:0.12em;">
                      📋 Storico classificazioni
                    </span>
                </div>""")
                history_html = gr.HTML(value=_history_table())

                classify_btn.click(
                    fn=classify_single,
                    inputs=[input_title, input_body],
                    outputs=[result_html, words_html, pri_words, history_html],
                )

            # ── TAB 2: BATCH ──
            with gr.TabItem("Batch CSV", id="batch"):
                gr.HTML("""
                <div style="text-align:center; padding:20px 0 12px;">
                  <h3 style="font-weight:800; font-size:1.2rem; margin:0;
                             letter-spacing:-0.02em;">Classificazione Batch</h3>
                  <p style="color:#64748B; font-size:0.82rem; margin-top:6px; font-weight:500;">
                    Carica un CSV con colonne <code style="padding:2px 6px;
                    border-radius:4px; font-size:0.78rem;">title</code> e
                    <code style="padding:2px 6px;
                    border-radius:4px; font-size:0.78rem;">body</code>
                  </p>
                </div>""")
                csv_input = gr.File(label="📄 Carica CSV", file_types=[".csv"], type="filepath")
                batch_btn = gr.Button("⚡ Classifica Batch", variant="primary")
                batch_preview = gr.HTML()
                csv_output = gr.File(label="📥 Scarica CSV con predizioni")
                batch_btn.click(fn=classify_batch, inputs=[csv_input], outputs=[batch_preview, csv_output])

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:20px 0 8px; color:#94A3B8; font-size:0.72rem;
                    font-weight:500; letter-spacing:0.05em;">
          © 2026 deskhero<br>
          Realizzato da Christian Cacchiotti

        </div>""")

    return app


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import google.colab; IS_COLAB = True
    except ImportError:
        IS_COLAB = False

    app = build_app()
    if IS_COLAB:
        print("\n" + "=" * 50)
        print("  🚀  Dashboard su Google Colab")
        print("=" * 50 + "\n")
        app.launch(share=True, quiet=False)
    else:
        print("\n" + "=" * 50)
        print("  🚀  http://localhost:7860")
        print("=" * 50 + "\n")
        app.launch(server_name="0.0.0.0", server_port=7860)
