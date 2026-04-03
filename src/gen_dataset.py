"""
=============================================================
 Deskhero - Accelerare il supporto aziendale tramite il triage automatico intelligente dei ticket
 Realizzato da Christian Cacchiotti
=============================================================

Genera un CSV di 1000 ticket sintetici per addestrare
un classificatore di categoria (Tecnico / Amministrazione / Commerciale) 
e priorità (alta / media / bassa).

Principi di design:
  ▸ Overlap lessicale tra categorie - la distinzione emerge
    dall'intento, non da parole esclusive;
  ▸ Priorità narrativa — l'urgenza emerge dal contesto del
    ticket e da segnali espliciti (keyword di urgenza/calma);
  ▸ Rumore realistico — errori ortografici, abbreviazioni,
    maiuscolo per frustrazione;
  ▸ Ticket cross-categoria — casi ambigui per robustezza;
  ▸ Ticket corti e generici — per favorire generalizzazione ed evitare overfitting;

Output: data/tickets.csv
"""

import csv, os, re, random
from collections import Counter

random.seed(42)

# ═══════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE
# ═══════════════════════════════════════════════════════════

EXTRA_SENTENCE_PROB    = 0.55   # prob. di aggiungere una frase extra di contesto al corpo del ticket
PRIORITY_SIGNAL_PROB   = 0.65   # prob. di iniettare un segnale esplicito di urgenza/calma
TITLE_NOISE_FACTOR     = 0.50   # riduce il rumore sul titolo rispetto al corpo (50% del livello body)
TYPO_PROB   = 0.60              # prob. di introdurre un errore ortografico per parola selezionata
ABBREV_PROB = 0.40              # prob. di abbreviare una parola (es. "perché" → "xké")
UPPER_PROB  = 0.20              # prob. di mettere una parola in MAIUSCOLO (simula frustrazione)
PUNCT_PROB  = 0.30              # prob. di aggiungere punteggiatura ripetuta (es. "!!!", "???")

PRIORITY_LABELS = ["alta", "media", "bassa"]   # etichette ordinali di priorità
PRIORITY_PROBS  = [0.25,   0.30,    0.45  ]    # distribuzione target realistica della priorità

# Distribuzione ticket per categoria: (n, noise_prob, noise_level)
DISTRIBUTION = {
    "Tecnico":         (320, 0.35, 0.50),
    "Amministrazione": (240, 0.15, 0.25),
    "Commerciale":     (190, 0.20, 0.30),
}
CROSS_CATEGORY_COUNT = 150       # cross-category narrativi
CROSS_CATEGORY_NOISE = (0.30, 0.40)
SHORT_CROSS_COUNT    = 50        # cross-category corti
SHORT_GENERIC_COUNT  = 50       # corti generici (vocabolario misto)

# Rumore sulle etichette di priorità: simula il disaccordo naturale
# tra operatori nella definizione della priorità (annotator disagreement).
# Il 10% dei ticket ha la priorità spostata di un livello.
LABEL_NOISE_PROB = 0.10

CLOSING_VARIATIONS = [
    "Resto a disposizione per chiarimenti.",
    "Grazie in anticipo per il riscontro.",
    "Cordiali saluti.",
    "Potete contattarmi al mio interno per approfondire.",
    "In attesa di un vostro riscontro.",
    "Rimango in attesa, grazie.",
    "Per qualsiasi chiarimento, sono disponibile.",
    "Vi ringrazio anticipatamente.",
    "Un saluto.",
    "Grazie per l'attenzione.",
    "Resto in attesa di aggiornamenti.",
    "Se servono dettagli aggiuntivi, sono reperibile via email.",
    "Ringrazio per la collaborazione.",
    "", "",  # nessuna chiusura (~13%)
]

# ═══════════════════════════════════════════════════════════
# 1b. SEGNALI DI PRIORITÀ
#     Iniettati nel corpo del ticket per fornire
#     al modello segnali lessicali coerenti con la priorità.
# ═══════════════════════════════════════════════════════════

PRIORITY_SIGNALS = {
    "alta": [
        "È una situazione urgente e bloccante.",
        "Urgente, il lavoro è completamente fermo.",
        "Priorità critica: ogni minuto perso ha un impatto diretto.",
        "È urgente, non possiamo procedere.",
        "Situazione bloccante, è urgente.",
        "È un problema critico, urgente.",
        "Urgente: rischiamo gravi conseguenze.",
        "Massima priorità, la situazione è insostenibile.",
        "Il problema è grave, va risolto urgentemente.",
        "Urgentissimo, il blocco è totale.",
        "È un'emergenza operativa.",
        "È critico, siamo in piena emergenza.",
    ],
    "media": [
        "Non è un'emergenza ma sarebbe opportuno intervenire a breve.",
        "La situazione è gestibile ma andrebbe risolta nei prossimi giorni.",
        "Rallenta il lavoro in modo significativo, anche se riesco a procedere.",
        "La situazione non è grave ma conviene non rimandare troppo.",
        "Riesco a procedere ma con difficoltà, andrebbe risolto a breve.",
        "Il problema è fastidioso, chiedo un controllo.",
        "Non è trascurabile, programmate un intervento.",
        "Vorrei una risoluzione entro la settimana se possibile.",
        "Il problema è moderato, riesco a lavorare con qualche disagio.",
        "Sta peggiorando, meglio intervenire presto.",
    ],
    "bassa": [
        "Non è prioritario, quando avete tempo.",
        "Nessuna premura, è solo una segnalazione.",
        "Bassa priorità, non crea problemi operativi.",
        "Lo segnalo per completezza, non è nulla di importante.",
        "Quando è comodo per voi, non c'è alcuna premura.",
        "Non è un problema grave, solo una segnalazione preventiva.",
        "È un dettaglio minore, da gestire quando avete un momento libero.",
        "Non c'è premura, il lavoro procede normalmente.",
        "Nessuna premura, è una richiesta informativa.",
        "È una cosa di poco conto, da risolvere con calma.",
        "Non è prioritario, lo segnalo solo per tenere traccia.",
        "Quando avete modo di verificare, con calma.",
    ],
}

# ═══════════════════════════════════════════════════════════
# 2. POOL LESSICALI CONDIVISE
# ═══════════════════════════════════════════════════════════

POOL = {
    "sistema":     ["gestionale SAP", "portale aziendale", "CRM", "sistema ERP",
                    "piattaforma intranet", "portale HR", "sistema di ticketing",
                    "software gestionale", "applicativo interno", "piattaforma documentale"],
    "software":    ["Microsoft Office", "Teams", "Outlook", "SAP GUI", "il gestionale",
                    "il software di contabilità", "Zoom", "il client VPN",
                    "l'applicativo HR", "Power BI"],
    "errore_msg":  ["Accesso negato", "Timeout di connessione", "Errore 500 - Internal Server Error",
                    "Credenziali non valide", "Servizio non disponibile", "File non trovato",
                    "Errore 403 - Forbidden", "Connessione rifiutata", "Memoria insufficiente",
                    "Sessione scaduta"],
    "tentativo":   ["riavviare il PC", "svuotare la cache", "reinstallare il programma",
                    "verificare le credenziali", "aggiornare il driver", "cambiare browser",
                    "contattare un collega", "disabilitare il firewall"],
    "numero_doc":  ["#2024-0871", "#FT-3392", "#ORD-1145", "#2024-0654", "#PED-8821",
                    "#CNT-0033", "#2025-0112", "#FT-4471", "#ORD-2287", "#2024-1093"],
    "data_ref":    ["15 ottobre", "3 novembre", "mese scorso", "fine settembre",
                    "12 dicembre", "7 gennaio", "fine ottobre", "22 novembre",
                    "4 febbraio", "metà marzo"],
    "scadenza":    ["fine mese", "questa settimana", "venerdì", "15 del mese prossimo",
                    "domani", "lunedì prossimo", "fine trimestre", "entro 48 ore"],
    "importo":     ["245,00", "1.320,50", "89,90", "3.750,00", "512,30",
                    "2.100,00", "430,75", "8.500,00", "155,20", "990,00"],
    "mese":        ["ottobre", "novembre", "dicembre", "settembre", "agosto",
                    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio"],
    "prodotto":    ["stampanti multifunzione", "licenze software", "notebook aziendali",
                    "monitor 27 pollici", "server rack", "servizi cloud",
                    "switch di rete", "scanner documentali", "telefoni VoIP", "tablet aziendali"],
    "dispositivo": ["HP LaserJet", "Canon iR", "Epson WorkForce", "la multifunzione del piano",
                    "Brother HL", "Ricoh SP"],
    "reparto":     ["vendite", "contabilità", "logistica", "risorse umane",
                    "acquisti", "marketing", "assistenza clienti", "direzione"],
    "cliente":     ["Rossi S.r.l.", "il cliente principale", "GreenTech S.p.A.",
                    "il fornitore Bianchi", "l'azienda partner", "il distributore regionale",
                    "MegaStore Italia", "lo studio Verdi"],
}

# ═══════════════════════════════════════════════════════════
# 3. TEMPLATES — PRIORITÀ CONTESTUALE
#    Ogni template ha TRE varianti (alta/media/bassa).
#    Il vocabolario è CONDIVISO: ciò che distingue la categoria
#    è l'INTENTO PRIMARIO del richiedente.
# ═══════════════════════════════════════════════════════════

TEMPLATES = {

    # ── TECNICO ─────────────────────────────────────────────
    # Intento: qualcosa NON FUNZIONA tecnicamente.
    "Tecnico": [
        {
            "title": ["Il {sistema} non permette di generare documenti",
                      "Errore durante la generazione documenti su {sistema}",
                      "Blocco emissione documenti dal {sistema}"],
            "alta":  "Da stamattina il {sistema} non permette di generare alcun documento. L'intero reparto {reparto} è bloccato, abbiamo decine di pratiche in scadenza oggi e il messaggio di errore è '{errore_msg}'. Ho già provato a {tentativo} senza alcun risultato.",
            "media": "Da un paio di giorni il {sistema} impiega tempi molto lunghi per generare i documenti. Riesco a completare il lavoro ma con rallentamenti evidenti. L'errore '{errore_msg}' compare saltuariamente.",
            "bassa": "Segnalo che il {sistema} talvolta mostra un avviso durante la generazione dei documenti. Non impedisce il lavoro e il documento viene prodotto regolarmente, ma vorrei capire la causa dell'avviso.",
        },
        {
            "title": ["Impossibile caricare ordini nel gestionale",
                      "Il portale non accetta nuovi ordini",
                      "Errore nel caricamento ordini su {sistema}"],
            "alta":  "Il {sistema} restituisce '{errore_msg}' quando si tenta di inserire un nuovo ordine. Il problema riguarda tutti gli operatori del reparto e nessun ordine può essere processato da stamattina. I clienti stanno già chiamando.",
            "media": "Alcuni ordini non vengono caricati correttamente nel {sistema}. Il problema è intermittente: su dieci tentativi circa tre falliscono con '{errore_msg}'. Riesco a procedere riprovando ma è poco efficiente.",
            "bassa": "Ho notato che il caricamento di nuovi ordini nel {sistema} è leggermente più lento del solito. Non ci sono errori visibili ma i tempi di risposta sono aumentati rispetto alla settimana scorsa.",
        },
        {
            "title": ["Connessione al server instabile",
                      "Problemi di rete che impediscono il lavoro",
                      "Rete aziendale non raggiungibile"],
            "alta":  "La connessione di rete è caduta completamente e l'intero piano è senza accesso ai servizi aziendali. Non possiamo consultare documenti, inviare email né processare alcuna richiesta. Serve intervento immediato.",
            "media": "La connessione di rete è instabile da ieri: funziona a intermittenza e la velocità è molto ridotta. Riesco a lavorare ma con continue interruzioni durante le videochiamate e il caricamento di file.",
            "bassa": "Da qualche giorno noto una leggera lentezza nella connessione di rete, soprattutto accedendo a {sistema}. Non crea veri problemi ma volevo segnalarlo per un controllo preventivo.",
        },
        {
            "title": ["{software} non si avvia dopo l'aggiornamento",
                      "Errore all'apertura di {software}",
                      "Crash di {software} dopo l'ultimo update"],
            "alta":  "{software} si chiude immediatamente dopo l'avvio con errore '{errore_msg}'. L'aggiornamento è stato installato ieri sera e da stamattina nessuno del reparto riesce ad usarlo. Abbiamo scadenze contabili che richiedono quel software oggi stesso.",
            "media": "{software} impiega diversi minuti per avviarsi e talvolta si blocca durante il salvataggio dei documenti. L'errore '{errore_msg}' compare una volta al giorno circa. Riesco a lavorare salvando più spesso.",
            "bassa": "{software} mostra un messaggio di avviso all'avvio ma funziona normalmente dopo averlo chiuso. È comparso dopo l'ultimo aggiornamento e vorrei sapere se è un comportamento noto.",
        },
        {
            "title": ["Stampante {dispositivo} non funzionante",
                      "{dispositivo} non risponde ai comandi",
                      "Impossibile stampare su {dispositivo}"],
            "alta":  "La stampante {dispositivo} non risponde a nessun comando e dobbiamo stampare la documentazione contrattuale per la riunione con {cliente} che è tra due ore. Nessuna delle altre stampanti del piano è disponibile.",
            "media": "La {dispositivo} stampa solo parzialmente i documenti, tagliando la parte destra di ogni foglio. I contratti risultano illeggibili. Per ora invio i documenti via email ma non è sempre possibile.",
            "bassa": "La qualità di stampa della {dispositivo} è peggiorata: le pagine escono con righe orizzontali chiare. Non è prioritario ma per i documenti ufficiali la presentazione non è accettabile.",
        },
        {
            "title": ["Accesso VPN bloccato da remoto",
                      "VPN aziendale non funzionante",
                      "Impossibile connettersi alla VPN"],
            "alta":  "Non riesco a connettermi tramite VPN da stamattina e lavoro esclusivamente da remoto. Ricevo '{errore_msg}' e non posso accedere a nessuna risorsa aziendale. Oggi ho una scadenza per la consegna del progetto al {reparto}.",
            "media": "La VPN si disconnette frequentemente, circa ogni mezz'ora. Riesco a riconnettermi ma perdo il lavoro non salvato e le videochiamate si interrompono. Ho provato a {tentativo} senza miglioramenti.",
            "bassa": "La VPN funziona ma la connessione è più lenta del solito, soprattutto accedendo a {sistema}. Non è un problema grave perché vado in ufficio quasi tutti i giorni, ma nei giorni di smart working è percepibile.",
        },
        {
            "title": ["Email aziendale non sincronizzata",
                      "Posta in arrivo non aggiornata",
                      "Problemi sincronizzazione email su {software}"],
            "alta":  "Le email non arrivano su {software} da stamattina. Ho verificato dal telefono e ci sono messaggi urgenti da {cliente} e dal reparto {reparto} che necessitano risposta entro oggi. Ho già provato a {tentativo}.",
            "media": "La sincronizzazione email su {software} avviene con ritardi di diverse ore. I messaggi arrivano sul telefono in tempo reale ma sul client desktop sono in ritardo. Ho provato a {tentativo} senza risultato.",
            "bassa": "Ogni tanto qualche email su {software} arriva con qualche minuto di ritardo rispetto al telefono. Non crea problemi concreti ma volevo segnalarlo per completezza.",
        },
        {
            "title": ["PC aziendale molto lento",
                      "Prestazioni degradate del computer",
                      "Lentezza anomala della postazione"],
            "alta":  "Il PC assegnatomi si blocca completamente durante l'elaborazione dei dati di fine mese. Il reparto {reparto} aspetta i report e io non riesco nemmeno ad aprire i file necessari. Serve una sostituzione o un intervento immediato.",
            "media": "Il PC è diventato molto lento: l'avvio richiede circa 10 minuti e passare da un'applicazione all'altra è frustrante. Riesco a lavorare ma la produttività ne risente parecchio.",
            "bassa": "Ho notato un leggero calo di prestazioni del PC nelle ultime settimane. {software} in particolare impiega qualche secondo in più per aprirsi. Per ora non è un problema ma potrebbe peggiorare.",
        },
        {
            "title": ["Backup automatico non eseguito",
                      "Mancata esecuzione del backup",
                      "Errore nel sistema di backup"],
            "alta":  "Il backup automatico non viene eseguito da tre giorni e ieri il server ha mostrato segni di instabilità. Se il disco dovesse cedere, perderemmo tutti i dati contabili e contrattuali del trimestre. L'errore nei log è '{errore_msg}'.",
            "media": "Il backup notturno non è andato a buon fine l'ultima notte. I log riportano '{errore_msg}'. I dati sono comunque disponibili sul server principale ma vorrei che venisse ripristinato il ciclo regolare.",
            "bassa": "Segnalo che il report del backup notturno di {sistema} mostra un avviso di tipo informativo. Il backup risulta completato ma con un tempo superiore alla media. Nessun dato sembra compromesso.",
        },
        {
            "title": ["Credenziali scadute, impossibile il reset",
                      "Password scaduta su {sistema}",
                      "Accesso negato al {sistema}: reset fallito"],
            "alta":  "Le mie credenziali di accesso al {sistema} sono scadute e il reset non funziona. Non ricevo l'email con il link e senza accesso non posso elaborare i pagamenti dei fornitori in scadenza oggi.",
            "media": "La password del {sistema} è scaduta. Il sistema di reset invia l'email ma il link risulta non valido. Un collega mi sta passando i dati urgenti ma non è una soluzione sostenibile.",
            "bassa": "Mi è arrivato l'avviso che la password del {sistema} scadrà tra 5 giorni. Ho provato a cambiarla in anticipo ma la procedura dà errore. Non è prioritario ma vorrei risolvere prima della scadenza.",
        },
    ],

    # ── AMMINISTRAZIONE ─────────────────────────────────────
    # Intento: il DOCUMENTO, il DATO o la PROCEDURA è errata.
    "Amministrazione": [
        {
            "title": ["Errore nell'importo della fattura {numero_doc}",
                      "Fattura {numero_doc} con importo non corretto",
                      "Discrepanza nella fattura {numero_doc}"],
            "alta":  "La fattura {numero_doc} riporta un importo di {importo} euro che non corrisponde a quanto concordato. La scadenza per il pagamento è {scadenza} e se non correggiamo prima rischiamo di pagare una cifra errata con conseguenze sul bilancio.",
            "media": "Ho riscontrato una discrepanza nell'importo della fattura {numero_doc}: risulta {importo} euro invece della cifra concordata. La scadenza non è imminente ma vorrei risolvere con anticipo per evitare complicazioni.",
            "bassa": "Segnalo una piccola incongruenza nella fattura {numero_doc}: una voce secondaria non corrisponde al listino. L'importo totale è sostanzialmente corretto e la scadenza è lontana, ma preferisco annotarlo.",
        },
        {
            "title": ["Anomalia nel cedolino paga di {mese}",
                      "Errore busta paga {mese}",
                      "Cedolino {mese} da verificare"],
            "alta":  "Il cedolino di {mese} mostra una trattenuta anomala di {importo} euro che non riconosco. L'importo netto accreditato è significativamente inferiore al dovuto e ho bisogno di chiarimenti immediati perché devo far fronte a pagamenti già programmati.",
            "media": "Ho notato che le ore di straordinario nel cedolino di {mese} non corrispondono al foglio presenze. La differenza è di alcune ore. Non è drammatico ma vorrei la correzione prima del prossimo cedolino.",
            "bassa": "Nel cedolino di {mese} la voce 'indennità di trasferta' non è presente nonostante la trasferta fosse stata approvata. Immagino sia un ritardo nella registrazione. Chiedo solo conferma.",
        },
        {
            "title": ["Contratto {numero_doc} in scadenza, rinnovo necessario",
                      "Scadenza imminente del contratto {numero_doc}",
                      "Rinnovo contratto {numero_doc}"],
            "alta":  "Il contratto {numero_doc} con {cliente} scade {scadenza} e non è ancora stato avviato il rinnovo. Se il contratto scade perdiamo le condizioni agevolate e il fornitore ha già comunicato che i prezzi aumenteranno.",
            "media": "Il contratto {numero_doc} è in scadenza tra circa un mese. Chiedo di avviare la procedura di rinnovo per evitare interruzioni del servizio. Non è imminente ma i tempi burocratici possono essere lunghi.",
            "bassa": "Segnalo che il contratto {numero_doc} scadrà tra qualche mese. Quando è il momento opportuno, vorrei capire se il rinnovo avverrà alle stesse condizioni o se serve rinegoziare.",
        },
        {
            "title": ["Rimborso spese trasferta del {data_ref}",
                      "Nota spese da liquidare",
                      "Richiesta rimborso per trasferta"],
            "alta":  "Ho presentato la nota spese per la trasferta del {data_ref} con un totale di {importo} euro e non ho ancora ricevuto il rimborso. Sono passate tre settimane e l'importo è rilevante per il mio bilancio personale. Ho già sollecitato due volte via email.",
            "media": "Allego la nota spese relativa alla trasferta del {data_ref} per un importo di {importo} euro. Chiedo il rimborso secondo le tempistiche previste dalle procedure aziendali.",
            "bassa": "Devo ancora presentare la nota spese della trasferta del {data_ref}. L'importo è modesto. Avrei bisogno del modulo aggiornato perché quello che ho sembra obsoleto.",
        },
        {
            "title": ["Bonifico {numero_doc} non risulta accreditato",
                      "Mancato accredito del pagamento {numero_doc}",
                      "Pagamento {numero_doc} non pervenuto"],
            "alta":  "Il bonifico {numero_doc} di {importo} euro del {data_ref} non risulta accreditato e il fornitore {cliente} ha inviato un sollecito formale minacciando di sospendere le consegne se non riceve conferma entro {scadenza}.",
            "media": "Il bonifico {numero_doc} del {data_ref} non risulta ancora accreditato al beneficiario. L'importo è di {importo} euro. Non ci sono conseguenze immediate ma vorrei capire se c'è stato un problema nella disposizione.",
            "bassa": "Chiedo conferma che il bonifico {numero_doc} del {data_ref} sia stato effettivamente disposto. Il beneficiario non ha segnalato problemi ma vorrei verificare per completezza documentale.",
        },
        {
            "title": ["Aggiornamento dati anagrafici nel sistema",
                      "Modifica dati personali",
                      "Rettifica informazioni anagrafiche"],
            "alta":  "Devo aggiornare urgentemente il mio indirizzo di residenza nel sistema perché la corrispondenza aziendale viene recapitata all'indirizzo vecchio. Ho già perso una comunicazione importante dal reparto {reparto}.",
            "media": "Chiedo di aggiornare il mio indirizzo di residenza nel sistema aziendale. Il trasferimento è avvenuto il mese scorso e vorrei che i dati fossero allineati.",
            "bassa": "Segnalo che il mio numero di telefono nel sistema aziendale non è aggiornato. Non crea problemi operativi ma vorrei correggerlo per uniformità.",
        },
        {
            "title": ["Nota di credito {numero_doc} non ricevuta",
                      "Mancata emissione nota di credito {numero_doc}",
                      "Attesa nota di credito per {numero_doc}"],
            "alta":  "La nota di credito relativa alla fattura {numero_doc} doveva essere emessa entro il {data_ref} ma non è ancora arrivata. Senza di essa non posso chiudere il bilancio trimestrale e la scadenza è {scadenza}.",
            "media": "Secondo gli accordi, la nota di credito per {numero_doc} doveva arrivare entro il {data_ref}. Non è ancora pervenuta. Non è critico nell'immediato ma la contabilità la richiede per la riconciliazione.",
            "bassa": "Chiedo aggiornamenti sulla nota di credito relativa a {numero_doc}. Nessuna premura ma vorrei tenere allineata la documentazione contabile.",
        },
        {
            "title": ["Errore nel calcolo ferie residue sul portale",
                      "Ferie residue non corrispondenti",
                      "Conteggio ferie errato nel {sistema}"],
            "alta":  "Il {sistema} mostra un residuo ferie errato: risultano molti meno giorni di quanti ne abbia effettivamente. Ho una ferie approvata la prossima settimana e temo che venga revocata se il dato non viene corretto.",
            "media": "Il conteggio delle ferie residue sul {sistema} non corrisponde ai miei calcoli. La differenza è di qualche giorno. Chiedo una verifica con i dati del foglio presenze.",
            "bassa": "Noto una piccola differenza nel conteggio ferie sul {sistema} rispetto a quanto mi aspettavo. Nessuna premura, potrebbe anche essere un mio errore di calcolo. Chiedo verifica quando possibile.",
        },
    ],

    # ── COMMERCIALE ─────────────────────────────────────────
    # Intento: RELAZIONE COMMERCIALE, ordine, cliente, trattativa.
    "Commerciale": [
        {
            "title": ["Preventivo per {prodotto} richiesto da {cliente}",
                      "Offerta commerciale per {prodotto}",
                      "Quotazione {prodotto} per nuovo cliente"],
            "alta":  "{cliente} ha richiesto un preventivo per {prodotto} e ci ha dato come termine massimo {scadenza}. Se non rispondiamo entro la data indicata, hanno già un'offerta alternativa pronta da un concorrente.",
            "media": "Vorremmo preparare un preventivo per {prodotto} richiesto da {cliente}. Non c'è una scadenza stringente ma il cliente è in fase di valutazione e una risposta rapida ci avvantaggerebbe.",
            "bassa": "Per un eventuale ordine futuro, {cliente} ha chiesto un'indicazione di prezzo su {prodotto}. Nessuna premura, stanno solo raccogliendo informazioni preliminari.",
        },
        {
            "title": ["Stato dell'ordine {numero_doc}",
                      "Aggiornamento sulla spedizione {numero_doc}",
                      "Ritardo nella consegna ordine {numero_doc}"],
            "alta":  "L'ordine {numero_doc} doveva essere consegnato il {data_ref} a {cliente}, che ha già minacciato di annullare se non riceve la merce entro {scadenza}. Il cliente rappresenta una quota significativa del nostro fatturato.",
            "media": "Vorrei un aggiornamento sull'ordine {numero_doc} effettuato il {data_ref}. Il cliente ha chiesto informazioni sulla data di consegna e vorrei poter dare una risposta.",
            "bassa": "Per completezza del mio report, vorrei sapere lo stato attuale dell'ordine {numero_doc}. Non ci sono solleciti da parte del cliente.",
        },
        {
            "title": ["Reclamo per merce danneggiata da {cliente}",
                      "Segnalazione danni su consegna a {cliente}",
                      "Contestazione merce ricevuta"],
            "alta":  "{cliente} ha ricevuto l'ordine {numero_doc} con materiale gravemente danneggiato e chiede sostituzione immediata o rimborso totale. Minacciano di interrompere il rapporto commerciale e rivolgersi alla concorrenza.",
            "media": "Alcuni articoli dell'ordine {numero_doc} sono arrivati con difetti di lieve entità a {cliente}. Il cliente chiede la sostituzione delle parti difettose. Il rapporto è buono ma conviene rispondere presto.",
            "bassa": "Un articolo minore dell'ordine {numero_doc} è arrivato con un'ammaccatura superficiale. Il cliente {cliente} non ha chiesto sostituzione ma preferisco annotare l'accaduto.",
        },
        {
            "title": ["Richiesta listino aggiornato per {cliente}",
                      "Invio nuovo catalogo prezzi necessario",
                      "Aggiornamento condizioni commerciali"],
            "alta":  "{cliente} deve chiudere un ordine importante entro {scadenza} ma il listino in suo possesso è datato {data_ref} e i prezzi non sono più validi. Senza il listino aggiornato rischiamo di perdere l'ordine.",
            "media": "Il listino prezzi attuale risale al {data_ref} e diversi clienti hanno chiesto la versione aggiornata. Vorrei poter distribuire il nuovo catalogo appena disponibile.",
            "bassa": "Per il mio archivio commerciale, potrei avere la versione aggiornata del listino prezzi di {prodotto}? Non ci sono trattative in corso che lo richiedano.",
        },
        {
            "title": ["Disponibilità {prodotto} per ordine urgente",
                      "Verifica stock {prodotto}",
                      "Giacenze {prodotto} richieste"],
            "alta":  "{cliente} vuole ordinare {prodotto} con consegna entro {scadenza} per un evento aziendale. Se non confermiamo la disponibilità entro oggi, si rivolgeranno altrove. L'ordine vale oltre {importo} euro.",
            "media": "Vorremmo sapere se {prodotto} è disponibile a magazzino. Abbiamo un paio di clienti interessati e mi servirebbe una conferma per procedere con le offerte.",
            "bassa": "Sto raccogliendo informazioni per la pianificazione del prossimo trimestre. Potreste indicarmi le giacenze attuali di {prodotto}?",
        },
        {
            "title": ["Richiesta sconto per ordine {numero_doc}",
                      "Trattativa prezzo con {cliente}",
                      "Condizioni speciali per ordine volume"],
            "alta":  "{cliente} condiziona un ordine da {importo} euro all'applicazione di uno sconto del 15%. La decisione deve arrivare entro {scadenza} altrimenti il budget verrà dirottato su un concorrente.",
            "media": "In considerazione del volume dell'ordine {numero_doc}, {cliente} ha chiesto condizioni più favorevoli. Sono in fase di negoziazione e un margine di sconto mi aiuterebbe a chiudere.",
            "bassa": "Per le prossime trattative con {cliente}, vorrei capire fino a che percentuale di sconto possiamo arrivare. Nessuna trattativa aperta al momento.",
        },
        {
            "title": ["Annullamento ordine {numero_doc}",
                      "Cancellazione ordine {numero_doc} su richiesta",
                      "Revoca ordine {numero_doc}"],
            "alta":  "{cliente} chiede l'annullamento immediato dell'ordine {numero_doc} del {data_ref} perché hanno trovato un errore critico nelle specifiche. Se la merce è già partita dobbiamo organizzare il rientro oggi stesso.",
            "media": "Per motivi interni, dobbiamo cancellare l'ordine {numero_doc} di {cliente}. La merce non dovrebbe ancora essere spedita. Chiedo conferma dell'annullamento e informazioni sull'eventuale penale.",
            "bassa": "{cliente} sta valutando se confermare o annullare l'ordine {numero_doc}. Per ora è solo un'ipotesi. Vorrei sapere quali sarebbero le condizioni di annullamento nel caso.",
        },
        {
            "title": ["Informazioni tecniche su {prodotto} per {cliente}",
                      "Scheda tecnica {prodotto} richiesta",
                      "Compatibilità {prodotto} con sistema del cliente"],
            "alta":  "{cliente} deve decidere entro {scadenza} e gli mancano le specifiche tecniche di {prodotto}. Hanno provato a consultare il {sistema} ma le informazioni non sono aggiornate. Se non rispondiamo oggi perdono la finestra di acquisto.",
            "media": "{cliente} chiede informazioni dettagliate su {prodotto}, in particolare sulle specifiche di compatibilità. Sono in fase di valutazione e un riscontro tempestivo ci farebbe fare buona impressione.",
            "bassa": "Per completare il nostro catalogo interno, avrei bisogno delle schede tecniche aggiornate di {prodotto}. Non ci sono richieste attive da parte dei clienti.",
        },
    ],
}

# ═══════════════════════════════════════════════════════════
# 3b. TEMPLATE CROSS-CATEGORIA (ticket ambigui)
# ═══════════════════════════════════════════════════════════

CROSS_CATEGORY_TEMPLATES = [
    {   # Tecnico/Amm → Tecnico (focus: malfunzionamento)
        "title": ["Il portale non calcola correttamente gli importi", "Errore di calcolo nel {sistema}"],
        "body": {
            "alta":  "Il {sistema} genera importi errati nei documenti contabili e l'intero reparto {reparto} è bloccato nell'elaborazione dei documenti di fine mese. Serve un fix immediato al modulo di calcolo.",
            "media": "Il {sistema} sembra calcolare in modo errato alcune voci nei documenti. Ho confrontato con i dati manuali e ci sono discrepanze. Riesco a correggere a mano ma è dispendioso.",
            "bassa": "Ho notato un arrotondamento insolito nei calcoli del {sistema}. La differenza è di pochi centesimi e potrebbe essere normale. Segnalo per un eventuale verifica.",
        },
        "category": "Tecnico",
    },
    {   # Amm/Comm → Amministrazione (focus: procedura)
        "title": ["Registrazione contabile dell'ordine {numero_doc}", "Documenti mancanti per ordine {numero_doc}"],
        "body": {
            "alta":  "L'ordine {numero_doc} di {cliente} non risulta registrato in contabilità nonostante la merce sia già stata consegnata. Se non registriamo entro {scadenza} ci saranno problemi con la chiusura del bilancio.",
            "media": "Manca la documentazione contabile per l'ordine {numero_doc}. La merce è stata spedita ma la fattura non è ancora stata emessa. Chiedo di procedere con la registrazione.",
            "bassa": "Per allineare i nostri archivi, segnalo che l'ordine {numero_doc} non ha ancora la corrispondente registrazione contabile. Non è prioritario ma conviene sistemare.",
        },
        "category": "Amministrazione",
    },
    {   # Comm/Tecnico → Commerciale (focus: cliente)
        "title": ["Cliente lamenta problemi con il nostro servizio", "Segnalazione di {cliente} su disservizio"],
        "body": {
            "alta":  "{cliente} ha segnalato che non riesce ad accedere al portale ordini e minaccia di passare alla concorrenza. Indipendentemente dalla causa tecnica, dobbiamo dare una risposta commerciale entro oggi per salvare il rapporto.",
            "media": "{cliente} ci ha contattato per lamentare rallentamenti nell'accesso al portale ordini. Il rapporto è buono ma conviene dare un riscontro per mantenere la fiducia.",
            "bassa": "Un cliente ha menzionato di aver avuto qualche difficoltà con il nostro portale online. Non ha fatto un reclamo formale, lo riporto per conoscenza.",
        },
        "category": "Commerciale",
    },
    {   # Tecnico/Comm → Tecnico (focus: bug nel sistema)
        "title": ["Il {sistema} mostra prezzi errati ai clienti", "Bug nei prezzi visualizzati sul portale"],
        "body": {
            "alta":  "Il {sistema} espone ai clienti prezzi che non corrispondono al listino aggiornato. Alcuni clienti stanno già ordinando a prezzi sbagliati. Se non correggiamo subito avremo un problema di ricavi significativo.",
            "media": "Ho notato che il {sistema} mostra alcuni prezzi non aggiornati nella sezione catalogo. Per ora nessun cliente ha ordinato a prezzi errati ma la situazione va corretta.",
            "bassa": "Segnalo che un paio di prodotti nel {sistema} mostrano un prezzo leggermente diverso dal listino corrente. La differenza è minima e potrebbe essere un ritardo di aggiornamento.",
        },
        "category": "Tecnico",
    },
    {   # Amm/Tecnico → Amministrazione (focus: dato)
        "title": ["Dati dei fornitori obsoleti nel sistema", "Anagrafica fornitori da aggiornare"],
        "body": {
            "alta":  "Le coordinate bancarie di {cliente} nel {sistema} sono errate e un pagamento di {importo} euro è stato respinto. Il fornitore ha già segnalato il mancato incasso e rischiamo penali per ritardato pagamento.",
            "media": "Diversi dati anagrafici dei fornitori nel {sistema} non sono aggiornati: indirizzi, PEC, codici fiscali. Questo crea problemi nella generazione automatica dei documenti. Chiedo un ciclo di aggiornamento.",
            "bassa": "Noto che alcune anagrafiche dei fornitori nel {sistema} hanno campi incompleti. Non crea problemi operativi ma sarebbe bene uniformare.",
        },
        "category": "Amministrazione",
    },
    {   # Comm/Amm → Commerciale (focus: trattativa)
        "title": ["Condizioni di pagamento per {cliente}", "Negoziazione termini di pagamento"],
        "body": {
            "alta":  "{cliente} condiziona un ordine da {importo} euro alla concessione di pagamento a 90 giorni anziché 30. La decisione serve entro {scadenza} o perdiamo l'affare. Servono autorizzazione amministrativa e valutazione rischio.",
            "media": "{cliente} ha chiesto la possibilità di pagamento dilazionato per il prossimo ordine. Vorrei capire se possiamo offrire condizioni diverse da quelle standard.",
            "bassa": "Per mia informazione, quali sono le condizioni di pagamento standard che offriamo ai nuovi clienti per {prodotto}? Sto preparando una presentazione commerciale generica.",
        },
        "category": "Commerciale",
    },
    {   # Tecnico/Amm — documenti + software
        "title": ["Il software non genera i report correttamente", "Errore generazione report dal {sistema}"],
        "body": {
            "alta":  "Il {sistema} non riesce a generare report da stamattina. Il reparto {reparto} è fermo e le scadenze sono imminenti.",
            "media": "Il {sistema} genera alcune report con dati incompleti. Riesco a correggerle manualmente ma richiede tempo.",
            "bassa": "Ho notato che il {sistema} salva i report in un formato leggermente diverso dal solito. Non crea problemi ma lo segnalo.",
        },
        "category": "Tecnico",
    },
    {   # Amm/Comm — ordine + contabilità
        "title": ["Ordine {numero_doc} senza copertura contabile", "Pagamento ordine {numero_doc} bloccato"],
        "body": {
            "alta":  "L'ordine {numero_doc} è stato confermato al cliente ma il budget non è stato approvato dalla contabilità. Il cliente aspetta la consegna entro {scadenza}.",
            "media": "L'ordine {numero_doc} necessita di approvazione contabile prima della spedizione. Il cliente non ha ancora sollecitato.",
            "bassa": "Per la registrazione dell'ordine {numero_doc}, mancano alcuni dati contabili. Non è prioritario.",
        },
        "category": "Amministrazione",
    },
    {   # Comm/Tecnico — portale clienti
        "title": ["Portale clienti lento", "Disservizio sul portale ordini online"],
        "body": {
            "alta":  "Diversi clienti ci stanno chiamando perché il portale ordini è inaccessibile. Stiamo perdendo ordini e la reputazione ne risente.",
            "media": "Il portale ordini online è più lento del solito. Alcuni clienti hanno segnalato tempi di caricamento lunghi.",
            "bassa": "Un cliente ha menzionato che il portale a volte è lento. Non abbiamo avuto altri reclami.",
        },
        "category": "Commerciale",
    },
    {   # Tecnico/Comm — listino nel gestionale
        "title": ["Listino prezzi non aggiornato nel {sistema}", "Prezzi errati sul gestionale"],
        "body": {
            "alta":  "I commerciali stanno inviando offerte con prezzi sbagliati perché il {sistema} non è aggiornato. Abbiamo già ricevuto contestazioni da {cliente}.",
            "media": "Il listino nel {sistema} è fermo al mese scorso. I commerciali chiedono l'aggiornamento per poter quotare correttamente.",
            "bassa": "Alcuni prezzi nel {sistema} sembrano non allineati al listino cartaceo. La differenza è minima.",
        },
        "category": "Tecnico",
    },
]

# Template corti cross-categoria (vocabolario misto tra reparti)
SHORT_CROSS_TEMPLATES = [
    {"title": "Fattura bloccata dal sistema", "body": "Il {sistema} non permette di registrare il documento {numero_doc}.", "category": "Tecnico"},
    {"title": "Errore contabile nel gestionale", "body": "Il gestionale calcola importi errati nei calcoli.", "category": "Tecnico"},
    {"title": "Ordine senza fattura", "body": "L'ordine {numero_doc} è stato spedito ma manca la fattura.", "category": "Amministrazione"},
    {"title": "Cliente non riesce a ordinare", "body": "Il portale ordini dà errore quando {cliente} prova a comprare.", "category": "Commerciale"},
    {"title": "Pagamento rifiutato dal sistema", "body": "Il bonifico per {cliente} è stato rifiutato, IBAN errato nel {sistema}.", "category": "Amministrazione"},
    {"title": "Prezzi sbagliati sul portale", "body": "I clienti vedono prezzi diversi dal listino sul portale.", "category": "Tecnico"},
    {"title": "Nota di credito per reso", "body": "Il cliente ha restituito la merce, serve la nota di credito.", "category": "Amministrazione"},
    {"title": "Contratto con errori", "body": "Il contratto generato dal {sistema} ha dati sbagliati del cliente.", "category": "Commerciale"},
    {"title": "Fornitore lamenta ritardo", "body": "Il fornitore non ha ricevuto il pagamento della fattura {numero_doc}.", "category": "Amministrazione"},
    {"title": "Software contabilità bloccato", "body": "Il software di contabilità non si apre, dà errore {errore_msg}.", "category": "Tecnico"},
    {"title": "Report vendite errato", "body": "Il report generato dal {sistema} non corrisponde ai dati commerciali.", "category": "Commerciale"},
    {"title": "Accesso negato al portale fornitori", "body": "Non riesco ad accedere al portale fornitori per caricare i documenti.", "category": "Tecnico"},
]

# Template corti generici — la categoria è assegnata casualmente
# perché il testo è troppo generico per determinare il reparto.
# Questo forza il classificatore a gestire l'ambiguità.
SHORT_GENERIC_TEMPLATES = [
    {"title": "Problema con il sistema", "body": "Il {sistema} dà un errore."},
    {"title": "Richiesta informazioni", "body": "Avrei bisogno di informazioni su una pratica."},
    {"title": "Aggiornamento dati", "body": "I dati nel {sistema} vanno aggiornati."},
    {"title": "Documento mancante", "body": "Manca un documento per la pratica {numero_doc}."},
    {"title": "Comunicazione", "body": "Devo inviare una comunicazione a {cliente}."},
    {"title": "Errore nel report", "body": "Il report del {sistema} ha dati errati."},
    {"title": "Problema accesso", "body": "Non riesco ad accedere al {sistema}."},
    {"title": "Verifica dati", "body": "Potete verificare i dati della pratica {numero_doc}?"},
    {"title": "Richiesta cliente", "body": "{cliente} ha fatto una richiesta."},
    {"title": "Scadenza in arrivo", "body": "C'è una scadenza per la pratica {numero_doc}."},
    {"title": "Problema con un ordine", "body": "L'ordine {numero_doc} ha un problema."},
    {"title": "Sistema lento", "body": "Il {sistema} è molto lento oggi."},
    {"title": "Modifica dati", "body": "Serve modificare dei dati nel {sistema}."},
    {"title": "Segnalazione", "body": "Segnalo un problema con {software}."},
    {"title": "Contatto cliente", "body": "Il cliente ha chiamato per un problema."},
]

# ═══════════════════════════════════════════════════════════
# 4. FRASI EXTRA (cross-vocabulary)
# ═══════════════════════════════════════════════════════════

EXTRA_SENTENCES = {
    "Tecnico": [
        "Il reparto {reparto} ha segnalato lo stesso problema.",
        "Anche {cliente} ci ha contattato per un disservizio simile.",
        "Il problema sta impattando la generazione dei documenti contabili.",
        "Allego lo screenshot dell'errore per facilitare la diagnosi.",
        "Il problema è comparso dopo l'ultimo aggiornamento di sistema.",
        "La situazione sta creando ritardi sugli ordini in lavorazione.",
        "I colleghi del {reparto} confermano lo stesso comportamento.",
    ],
    "Amministrazione": [
        "Il {sistema} registra correttamente il dato, il problema è nel contenuto.",
        "Ho verificato anche con il reparto {reparto} e confermano l'anomalia.",
        "Il problema potrebbe avere impatti sulla operatività del reparto.",
        "Allego la documentazione a supporto della richiesta.",
        "La questione è legata a obblighi normativi in scadenza.",
        "{cliente} ha fatto presente la stessa discrepanza.",
        "Chiedo gentilmente di essere aggiornato sullo stato della pratica.",
    ],
    "Commerciale": [
        "Ho già verificato sul {sistema} ma le informazioni sono incomplete.",
        "Il reparto {reparto} è in copia per conoscenza.",
        "La documentazione contrattuale è stata inviata al cliente.",
        "Stiamo valutando anche offerte alternative sul mercato.",
        "Il budget del cliente è già approvato e sono pronti a procedere.",
        "Confidiamo in una risposta rapida per procedere con l'ordine.",
        "Il reparto {reparto} ha confermato la disponibilità tecnica.",
    ],
}

# ═══════════════════════════════════════════════════════════
# 5. MOTORE DI RUMORE
# ═══════════════════════════════════════════════════════════

ERRORI_COMUNI = {
    "problema": ["prblema", "problama"], "sistema": ["ssitema", "sistama"],
    "connessione": ["conessione", "connessoine"], "fattura": ["fatua", "fattua"],
    "accesso": ["acesso", "acceso"], "aggiornamento": ["agiorrnamento", "aggiornamneto"],
    "disponibilità": ["disponibilita", "disponiblità"], "consegna": ["consena", "consegnga"],
}

ABBREVIAZIONI = {
    "non riesco": "nn riesco", "per favore": "x favore", "comunque": "cmq",
    "probabilmente": "prob", "informazioni": "info", "grazie": "grz",
}


def inject_noise(text: str, noise_level: float = 0.0) -> str:
    """Applica rumore ortografico, abbreviazioni, maiuscolo e punteggiatura."""
    if noise_level == 0.0:
        return text
    if random.random() < noise_level * TYPO_PROB:
        for corretto, sbagliati in ERRORI_COMUNI.items():
            if corretto in text.lower():
                text = re.sub(rf"\b{re.escape(corretto)}\b",
                              random.choice(sbagliati), text, count=1, flags=re.IGNORECASE)
    if random.random() < noise_level * ABBREV_PROB:
        for esteso, abbrev in ABBREVIAZIONI.items():
            if esteso in text.lower():
                text = re.sub(rf"\b{re.escape(esteso)}\b",
                              abbrev, text, count=1, flags=re.IGNORECASE)
    if random.random() < noise_level * UPPER_PROB:
        text = text.upper()
    if random.random() < noise_level * PUNCT_PROB:
        text = text.rstrip(".") + random.choice(["!!!", "!!", ".", ""])
    return text


def fill_template(template: str) -> str:
    """Sostituisce ogni {chiave} con un valore casuale dalla pool."""
    return re.sub(r"\{(\w+)\}", lambda m: random.choice(POOL[m.group(1)])
                  if m.group(1) in POOL else m.group(0), template)


# ═══════════════════════════════════════════════════════════
# 6. GENERAZIONE
# ═══════════════════════════════════════════════════════════

def _build_ticket(templates, category, noise_prob, noise_level, ticket_id):
    """Costruisce un singolo ticket standard."""
    priority = random.choices(PRIORITY_LABELS, weights=PRIORITY_PROBS)[0]
    tmpl = random.choice(templates)
    title_src = random.choice(tmpl["title"]) if isinstance(tmpl["title"], list) else tmpl["title"]
    body = tmpl[priority]

    extra = EXTRA_SENTENCES.get(category, [])
    if extra and random.random() < EXTRA_SENTENCE_PROB:
        body = body.rstrip() + " " + random.choice(extra)

    # Segnale esplicito di priorità (urgenza / non-urgenza)
    if random.random() < PRIORITY_SIGNAL_PROB:
        body = body.rstrip() + " " + random.choice(PRIORITY_SIGNALS[priority])

    title, body = fill_template(title_src), fill_template(body)
    closing = random.choice(CLOSING_VARIATIONS)
    if closing:
        body = body.rstrip() + " " + closing

    nl = noise_level if random.random() < noise_prob else 0.0
    title = inject_noise(title, nl * TITLE_NOISE_FACTOR)
    body  = inject_noise(body, nl)

    return {"id": ticket_id, "title": title.strip(), "body": body.strip(),
            "category": category, "priority": priority}


def _build_cross_ticket(ticket_id):
    """Costruisce un singolo ticket cross-categoria."""
    tmpl = random.choice(CROSS_CATEGORY_TEMPLATES)
    priority = random.choices(PRIORITY_LABELS, weights=PRIORITY_PROBS)[0]
    category = tmpl["category"]

    title_src = random.choice(tmpl["title"]) if isinstance(tmpl["title"], list) else tmpl["title"]
    body = tmpl["body"][priority]

    extra = EXTRA_SENTENCES.get(category, [])
    if extra and random.random() < EXTRA_SENTENCE_PROB:
        body = body.rstrip() + " " + random.choice(extra)

    # Segnale esplicito di priorità
    if random.random() < PRIORITY_SIGNAL_PROB:
        body = body.rstrip() + " " + random.choice(PRIORITY_SIGNALS[priority])

    title, body = fill_template(title_src), fill_template(body)
    closing = random.choice(CLOSING_VARIATIONS)
    if closing:
        body = body.rstrip() + " " + closing

    np_, nl = CROSS_CATEGORY_NOISE
    nl_eff = nl if random.random() < np_ else 0.0
    title = inject_noise(title, nl_eff * TITLE_NOISE_FACTOR)
    body  = inject_noise(body, nl_eff)

    return {"id": ticket_id, "title": title.strip(), "body": body.strip(),
            "category": category, "priority": priority}


def _build_short_cross_ticket(ticket_id):
    """Ticket corto cross-categoria: vocabolario misto tra reparti."""
    tmpl = random.choice(SHORT_CROSS_TEMPLATES)
    priority = random.choices(PRIORITY_LABELS, weights=PRIORITY_PROBS)[0]
    category = tmpl["category"]
    title_src = tmpl["title"]
    body = tmpl["body"]

    if random.random() < PRIORITY_SIGNAL_PROB:
        body = body.rstrip() + " " + random.choice(PRIORITY_SIGNALS[priority])

    title, body = fill_template(title_src), fill_template(body)
    nl = 0.45 if random.random() < 0.40 else 0.0
    title = inject_noise(title, nl * TITLE_NOISE_FACTOR)
    body = inject_noise(body, nl)
    return {"id": ticket_id, "title": title.strip(), "body": body.strip(),
            "category": category, "priority": priority}


def _build_short_generic_ticket(ticket_id):
    """Ticket corto generico: vocabolario misto, categoria assegnata
    casualmente — simula ticket reali troppo vaghi per essere classificati
    con certezza. Introduce rumore nella classificazione per categoria."""
    tmpl = random.choice(SHORT_GENERIC_TEMPLATES)
    priority = random.choices(PRIORITY_LABELS, weights=PRIORITY_PROBS)[0]
    category = random.choice(["Tecnico", "Amministrazione", "Commerciale"])
    title_src = tmpl["title"]
    body = tmpl["body"]

    if random.random() < PRIORITY_SIGNAL_PROB:
        body = body.rstrip() + " " + random.choice(PRIORITY_SIGNALS[priority])

    title, body = fill_template(title_src), fill_template(body)
    nl = 0.40 if random.random() < 0.35 else 0.0
    title = inject_noise(title, nl * TITLE_NOISE_FACTOR)
    body = inject_noise(body, nl)
    return {"id": ticket_id, "title": title.strip(), "body": body.strip(),
            "category": category, "priority": priority}


def generate_dataset():
    """Genera l'intero dataset di 1000 ticket sintetici."""
    tickets, tid = [], 1
    for cat, (n, np_, nl) in DISTRIBUTION.items():
        for _ in range(n):
            tickets.append(_build_ticket(TEMPLATES[cat], cat, np_, nl, tid))
            tid += 1
    for _ in range(CROSS_CATEGORY_COUNT):
        tickets.append(_build_cross_ticket(tid))
        tid += 1
    for _ in range(SHORT_CROSS_COUNT):
        tickets.append(_build_short_cross_ticket(tid))
        tid += 1
    for _ in range(SHORT_GENERIC_COUNT):
        tickets.append(_build_short_generic_ticket(tid))
        tid += 1
    random.shuffle(tickets)

    # Label noise: simula il disaccordo naturale tra operatori
    # nella definizione della priorità (~10% dei ticket).                                                    
    # La priorità viene spostata di un livello (alta↔media, media↔bassa).
    SHIFT = {"alta": "media", "media": random.choice(["alta", "bassa"]), "bassa": "media"}
    noise_count = 0
    for t in tickets:
        if random.random() < LABEL_NOISE_PROB:
            old = t["priority"]
            t["priority"] = random.choice(["alta", "bassa"]) if old == "media" else "media"
            noise_count += 1

    for i, t in enumerate(tickets, 1):
        t["id"] = i
    return tickets


# ═══════════════════════════════════════════════════════════
# 7. EXPORT CSV + STATISTICHE IN OUTPUT
# ═══════════════════════════════════════════════════════════

def export_csv(tickets, output_path):
    """Esporta il dataset in formato CSV."""
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "body", "category", "priority"])
        w.writeheader()
        w.writerows(tickets)
    print(f"[OK] Dataset salvato in: {output_path}")


def print_stats(tickets):
    """Stampa statistiche descrittive del dataset generato."""
    n = len(tickets)
    cat_c = Counter(t["category"] for t in tickets)
    pri_c = Counter(t["priority"] for t in tickets)
    cross = Counter((t["category"], t["priority"]) for t in tickets)

    print(f"\n{'═' * 55}")
    print(f"  STATISTICHE DATASET  ()")
    print(f"{'═' * 55}")
    print(f"  Totale ticket:  {n}\n")

    print("  Distribuzione per CATEGORIA:")
    for cat, c in sorted(cat_c.items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {c:>4}  ({c/n*100:4.1f}%)  {'█'*(c//10)}")

    print("\n  Distribuzione per PRIORITÀ:")
    for pri in PRIORITY_LABELS:
        c = pri_c[pri]
        print(f"    {pri:<20} {c:>4}  ({c/n*100:4.1f}%)  {'█'*(c//10)}")

    print("\n  Cross-tab CATEGORIA × PRIORITÀ:")
    cats = sorted(cat_c.keys())
    print(f"  {'':20}" + "".join(f"{p:>8}" for p in PRIORITY_LABELS))
    for cat in cats:
        print(f"  {cat:<20}" + "".join(f"{cross[(cat, p)]:>8}" for p in PRIORITY_LABELS))

    unique = len({t["body"] for t in tickets})
    pct = unique / n * 100
    print(f"\n  Unicità corpi: {unique}/{n} ({pct:.1f}%)")

    print(f"{'═' * 55}\n")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tickets.csv")
    print("Generazione dataset  in corso...")
    tickets = generate_dataset()
    print_stats(tickets)
    export_csv(tickets, OUTPUT_PATH)
