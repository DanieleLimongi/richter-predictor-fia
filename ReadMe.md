# Richter Predictor - Sistema di Previsione dei Danni da Terremoti

Il Richter Predictor è un sistema di machine learning progettato per predire il livello di danneggiamento degli edifici causato da eventi sismici. Utilizzando come caso studio il terremoto devastante del Nepal del 2015, il progetto affronta una sfida critica nell'ambito della gestione dei disastri e della pianificazione urbana: stimare rapidamente e accuratamente i danni strutturali su larga scala per ottimizzare gli interventi di soccorso e la ricostruzione. Il sistema combina tecniche avanzate di preprocessamento dei dati e machine learning per fornire previsioni affidabili e scalabili.

## Struttura del Progetto

```
richter-predictor-fia/
├── config/
│   └── dataset_config.json          # Configurazione centrale della pipeline
├── data/
│   ├── raw/                         # Dati CSV originali
│   │   ├── train_values.csv         # Feature degli edifici
│   │   └── train_labels.csv         # Target (damage_grade)
│   └── interim_tf/                  # Dati preprocessati
│       ├── train_interim.parquet    # Dataset in formato Parquet
│       ├── train_interim.tfrecord   # Dataset per TensorFlow
│       └── dtype_mapping.json       # Metadati dei tipi di dati
├── src/
│   ├── features/
│   │   ├── categorical_threshold_search.py  # Ottimizzazione soglie
│   │   └── build_tf_pipeline.py             # Costruzione modello TF
│   └── data/
│       └── make_dataset_tf.py       # Generazione dataset preprocessato
├── models/                          # Modelli salvati
├── reports/                         # Risultati ottimizzazione
└── logs/                           # Log del training
```

## Requisiti del Sistema

### Dipendenze Software
- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Scikit-learn
- Librerie specificate in `requirements.txt`

### Installazione
```bash
git clone [https://github.com/DanieleLimongi/richter-predictor-fia]
cd richter-predictor-fia
pip install -r requirements.txt
```

## Pipeline di Preprocessamento
La pipeline del progetto è progettata per trasformare i dati grezzi in un formato ottimizzato per il training dei modelli TensorFlow. Ogni fase della pipeline è strettamente integrata e coordinata, garantendo che i dati vengano preprocessati in modo coerente e che il modello finale sia costruito su una base solida. La pipeline parte dalla configurazione del dataset, passa per l'ottimizzazione delle soglie delle feature categoriche, la generazione del dataset preprocessato e termina con la costruzione della pipeline TensorFlow.

### Configurazione del Dataset

Il file `dataset_config.json` è il punto di riferimento centrale per la configurazione del preprocessamento dei dati. Questo file JSON contiene tutte le informazioni necessarie per garantire che gli script della pipeline lavorino in modo coerente e coordinato. La sua funzione principale è quella di centralizzare i parametri chiave, evitando che vengano hardcoded nei singoli script, e permettendo una facile modifica e gestione delle impostazioni.

La configurazione include i percorsi dei dati grezzi e preprocessati, le soglie per le feature categoriche e i parametri per la cross-validation. Ad esempio, i percorsi dei file di input (`train_values.csv` e `train_labels.csv`) e delle directory di output (`interim_tf`) sono definiti chiaramente, garantendo che gli script sappiano dove trovare i dati e dove salvare i risultati. Inoltre, le soglie per le feature categoriche `geo_level_2_id` e `geo_level_3_id` possono essere configurate manualmente o ottimizzate automaticamente. Questo approccio offre flessibilità, permettendo di passare facilmente da un metodo manuale a uno automatico per la gestione delle soglie.

### Ottimizzazione delle Soglie Categoriali
Lo script `categorical_threshold_search.py` è stato sviluppato per affrontare il problema delle feature categoriche ad alta cardinalità, come `geo_level_2_id` e `geo_level_3_id`. Queste feature, caratterizzate da migliaia di categorie uniche, possono causare inefficienze computazionali, problemi di memoria e overfitting nei modelli di machine learning. Per risolvere questa sfida, lo script implementa un processo di ottimizzazione delle soglie che consente di raggruppare le categorie rare sotto un valore comune chiamato "OTHER" (codificato come `-1`), riducendo la cardinalità delle feature senza compromettere le performance del modello.

Il processo inizia con il caricamento dei dati grezzi, che vengono uniti per creare un dataset completo. Successivamente, lo script applica soglie di frequenza percentuale alle feature categoriche, trasformando le categorie con frequenza inferiore alla soglia in "OTHER". Questo passaggio semplifica la struttura dei dati, riducendo la complessità del modello e migliorandone la scalabilità. Per identificare le soglie ottimali, lo script utilizza un approccio basato su **Random Search**, supportato da tecniche di validazione come **K-Fold semplice** e **Nested Cross-Validation**.

Il **Random Search** rappresenta il cuore dello script. Genera configurazioni casuali di soglie per le feature `geo_level_2_id` e `geo_level_3_id`, applicandole al dataset per trasformare le categorie rare. Tuttavia, il Random Search da solo non è sufficiente per determinare la qualità delle soglie generate. Per farlo, è necessario un modello di machine learning, come il **RandomForestClassifier**, e una tecnica di validazione che stimi le performance del modello con le soglie applicate.

Le tecniche di validazione svolgono un ruolo cruciale nel processo di ottimizzazione. Il **K-Fold semplice** divide il dataset in più fold, addestra il modello su una parte del dataset e lo valuta sulla parte rimanente. Questo approccio calcola una media delle performance su tutti i fold, fornendo una stima rapida e affidabile della qualità delle soglie. La **Nested Cross-Validation**, invece, combina un outer loop per la valutazione generale e un inner loop per l'ottimizzazione degli iperparametri del modello. L'outer loop divide il dataset in fold per stimare le performance generali delle soglie, mentre l'inner loop utilizza una ricerca su griglia per ottimizzare parametri come il numero di alberi e la profondità massima del modello. Questo approccio è più robusto rispetto al K-Fold semplice, poiché separa il processo di ottimizzazione degli iperparametri dalla valutazione finale.

In sintesi, il Random Search esplora lo spazio delle soglie, generando configurazioni casuali che vengono applicate al dataset, mentre il K-Fold e la Nested Cross-Validation valutano la qualità delle soglie applicate stimando le performance del modello. Questo processo integrato garantisce che le soglie selezionate siano effettivamente utili per migliorare le performance del modello, riducendo la cardinalità delle feature senza perdere informazioni rilevanti. Alla fine, lo script calcola statistiche dettagliate sulle categorie originali e rimanenti, fornendo una visione chiara dell'impatto delle trasformazioni e dell'efficacia delle soglie ottimali.

### Generazione del Dataset
Lo script `make_dataset_tf.py` è responsabile della trasformazione dei dati grezzi in un formato ottimizzato per il training dei modelli TensorFlow. Si occupa di applicare le soglie ottimali alle feature categoriche, effettuare il casting dei tipi di dati e salvare il dataset preprocessato in formati compatibili con TensorFlow.

Il processo inizia con il caricamento dei dati grezzi e il loro merge per creare un dataset completo. Le soglie ottimali, trovate in `categorical_threshold_search.py`, vengono applicate alle feature `geo_level_2_id` e `geo_level_3_id` per ridurre la cardinalità. Lo script converte i tipi di dati per garantire compatibilità con TensorFlow e ottimizzare l'uso della memoria. Ad esempio, le feature numeriche vengono convertite in `float32`, le categoriche in `int32` e il target in `int8`.

Il dataset preprocessato viene salvato in due formati: **Parquet**, utile per analisi e debugging, e **TFRecord**, ottimizzato per TensorFlow. Inoltre, lo script salva un file JSON con i metadati del dataset, inclusi i tipi di dati e le soglie applicate, garantendo la riproducibilità della pipeline.

### Costruzione della Pipeline TensorFlow
Lo script `build_tf_pipeline.py` è progettato per costruire automaticamente una pipeline TensorFlow/Keras che integra il preprocessamento dei dati e il training del modello. Si basa sui dati preprocessati generati da `make_dataset_tf.py` e utilizza i metadati salvati (ad esempio, `dtype_mapping.json`) per configurare dinamicamente i layer di input e preprocessamento.

La pipeline gestisce le feature numeriche e categoriche con layer dedicati. Le feature numeriche vengono normalizzate per garantire stabilità nel training e velocizzare la convergenza, mentre le feature categoriche vengono trasformate in rappresentazioni vettoriali dense tramite embedding. La dimensione degli embedding è calcolata dinamicamente in base alla cardinalità della feature, garantendo che le rappresentazioni siano adeguate alla complessità dei dati.

L'architettura del modello è progettata per essere flessibile e adattarsi dinamicamente ai dati forniti. Include layer densi con Dropout e BatchNormalization per regolarizzazione e stabilità, e un output layer con softmax per la classificazione multi-classe. Il modello viene compilato con l'ottimizzatore Adam e un learning rate conservativo di 0.001, garantendo stabilità e velocità di convergenza.

## Come Utilizzare il Sistema

### Preparazione dei Dati
Posizionare i file `train_values.csv` e `train_labels.csv` nella directory `data/raw/`.

### Esecuzione della Pipeline Completa

#### 1. Ottimizzazione delle Soglie
```bash
python src/features/categorical_threshold_search.py
```
**Output**: Genera `reports/threshold_search_results.json` con le soglie ottimali.

**Risultati attesi**:
```
=== RISULTATI OTTIMIZZAZIONE ===
Miglior score: 0.7234
Soglia geo_level_2_id: 0.0157
Soglia geo_level_3_id: 0.0089

=== STATISTICHE CATEGORIE ===
geo_level_2_id:
  Categorie originali: 1427
  Categorie rimanenti: 45
  Riduzione: 96.8%
  Campioni 'OTHER': 2847 (11.2%)
```

#### 2. Preprocessamento del Dataset
```bash
python src/data/make_dataset_tf.py
```
**Output**: 
- `data/interim_tf/train_interim.parquet` (per analisi)
- `data/interim_tf/dtype_mapping.json` (metadati)

#### 3. Conversione Dataset in formato Keras
```bash
python src/features/build_tf_pipeline.py`
```
**Output**: 
- `models/preproc_tf.keras

## Risoluzione Problemi Comuni

### Errori di Memoria
- Ridurre `batch_size` in `build_tf_pipeline.py`
- Usare soglie più aggressive (valori più alti) per ridurre la cardinalità

### Performance Basse
- Verificare che le soglie siano state applicate correttamente
- Controllare la distribuzione delle classi nel target
- Aumentare il numero di iterazioni nel Random Search

### File non Trovati
- Verificare la struttura delle directory
- Controllare i percorsi in `dataset_config.json`
- Assicurarsi che i file CSV siano nella directory `data/raw/`

## Scelte Progettuali e Iperparametri

Le scelte progettuali e gli iperparametri del sistema sono stati attentamente selezionati per bilanciare accuratezza, efficienza e scalabilità. Le soglie per le feature categoriche sono state ottimizzate utilizzando Random Search e Nested Cross-Validation, garantendo che le trasformazioni riducano la cardinalità senza compromettere le performance. La dimensione degli embedding è calcolata dinamicamente per adattarsi alla cardinalità delle feature, mentre l'architettura della rete neurale include Dropout progressivo e BatchNormalization per prevenire overfitting e accelerare la convergenza.

## Conclusione

Il Richter Predictor rappresenta un sistema completo e scalabile per la predizione dei danni strutturali causati da eventi sismici. La pipeline di preprocessamento garantisce che i dati siano trasformati in modo coerente e ottimizzato, mentre la costruzione automatica della pipeline TensorFlow/Keras permette di addestrare modelli robusti e accurati. Grazie alla sua flessibilità e scalabilità, il sistema può essere facilmente adattato a diversi scenari e dataset, fornendo un supporto prezioso per la gestione dei disastri e la pianificazione urbana.