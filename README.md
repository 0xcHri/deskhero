<div align="center">
  <img src="https://raw.githubusercontent.com/0xcHri/deskhero/refs/heads/main/assets/img/deskhero%20logo.png#gh-light-mode-only" width="200">
  <img src="https://raw.githubusercontent.com/0xcHri/deskhero/refs/heads/main/assets/img/deskhero%20logo%202.png#gh-dark-mode-only" width="200">
</div>

<div align="center">
  
## Accelerare il supporto aziendale tramite il triage automatico intelligente dei ticket

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

Gestire manualmente decine o centinaia di ticket al giorno è un’attività ripetitiva, soggetta a errori e che rallenta significativamente il tempo di risposta al cliente.

Questo Project Work per il corso di Informatica per le Aziende Digitali (L-31) affronta proprio questo problema reale aziendale, sviluppando un sistema di triage automatico intelligente tramite tecniche di Machine Learning, che è in grado di:

* Riconoscere se un ticket riguarda l’area Amministrativa, Tecnica o Commerciale;
* Assegnare una priorità (Alta / Media / Bassa) in base al contenuto;
* Offrire un’interfaccia web semplice e immediata per classificare nuovi ticket con un click;

L'architettura sviluppata si fonda su una rappresentazione testuale TF-IDF con gestione della negazione italiana e stemming leggero, affiancata da feature manuali di urgenza.

Per la classificazione della categoria è stato adottato un approccio diretto a tre classi mediante Support Vector Machine lineare (LinearSVC), mentre per la priorità è stata implementata una decomposizione binaria a due stadi, ispirata alla metodologia di Frank e Hall (2001), che risolve il problema strutturale della classe intermedia. 

## Struttura del repository

```plaintext
deskhero/
├── data/
│   └── tickets.csv                  # Dataset in csv
├── models/
│   ├── svm_category.joblib          # Classificatore categoria
│   ├── svm_is_alta.joblib           # Stadio 1 priorità (alta vs non-alta)
│   ├── svm_is_bassa.joblib          # Stadio 2 priorità (bassa vs media)
│   ├── tfidf_vectorizer.joblib      # Vettorizzatore TF-IDF addestrato
│   └── metadata.json                # Metriche, configurazione, label
├── src/
│   ├── gen_dataset.py               # Generatore dataset sintetico
│   ├── ml_pipeline.py               # Pipeline ML completa
│   └── dashboard.py                 # Dashboard interattiva (Gradio)
├── assets/
│   └── img/                         # Immagini
├── requirements.txt                 # Dipendenze
├── README.md                        # README
└── LICENSE.md                       # Licenza MIT
```

## Installazione e Setup

E' possibile scegliere tra due modalità:

### 1. Google Colab (Modalità consigliata)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N7NV3X9O_hNU-C0wnjep7YCuuGzhLZoj)

**Istruzioni:**
1. Cliccare sul badge oppure aprire direttamente il notebook tramite [questo link.](https://colab.research.google.com/drive/1N7NV3X9O_hNU-C0wnjep7YCuuGzhLZoj).
2. Eseguire le celle in ordine (Runtime → Esegui tutte).
3. Il dataset verrà generato automaticamente.
4. I modelli verranno addestrati.
5. Alla fine si avvierà la dashboard interattiva.

> [!NOTE]
> La dashboard verrà resa pubblica e accessibile tramite un link temporaneo generato dall'esecuzione di dashboard.py

### 2. Esecuzione Locale

#### Installazione

```bash
git clone deskhero

pip install -r requirements.txt

# 1. Genera il dataset
python src/gen_dataset.py

# 2. Addestra i modelli e visualizza le comparazioni
python src/ml_pipeline.py

# 3. Avvia la dashboard su localhost:7860
python src/dashboard.py
```

## Licenza

Distribuito con licenza MIT. Vedi [LICENSE.md](LICENSE.md).
