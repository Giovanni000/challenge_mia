# AN2DL25 Challenge 1 — Nota di Lavoro

## Obiettivo del progetto
- Classificare sequenze multivariate (180 time step, 37 feature) del dataset `Pirate Pain` in tre classi: `no_pain`, `low_pain`, `high_pain`.
- Vietato l'uso di modelli pre-addestrati: l'intera pipeline deve allenare da zero architetture ricorrenti.
- Produrre un notebook eseguibile (`challenge1_solution.ipynb`) e un file di submission `.csv` per Kaggle.

## Collegamenti con il notebook di lezione
- **Notebook di riferimento:** `old_notebooks/Timeseries Classification (1).ipynb`, orientato a Google Colab e dataset dimostrativi.
- **Principali differenze introdotte:**
  - Gestione automatica di path locali/Colab, mount di Google Drive e directory dedicate `outputs/`, `logs/`, `checkpoints/`.
  - Pipeline dati adattata al formato long del Pirate Pain (pivot → tensori `(N, T, F)`), encoding delle feature categoriche e normalizzazione globale.
  - Implementazione PyTorch modulare (`RecurrentBackbone`) con supporto a RNN/GRU/LSTM, versioni bidirezionali e hyperparam personalizzabili.
  - Training loop avanzato (`fit_model`) con mixed precision, gradient clipping, ReduceLROnPlateau monitorato su macro-F1, early stopping configurabile, checkpointing e logging TensorBoard.
  - Sistema di esperimenti (`prepare_config`, `run_experiment`) per lanciare più configurazioni in sequenza, salvare i risultati e confrontarli tramite `summary_table`.

## Adattamento al dataset Pirate Pain
- **Pre-processing:**
  - Pivot per ottenere sequenze ordinate per `sample_index` e `time` (180 step × 37 feature).
  - Normalizzazione feature-wise usando media/deviazione dello split di training.
  - Codifica di `n_legs`, `n_hands`, `n_eyes` in valori interi condivisi con il test set.
  - Mapping etichette stringa → indici (`LABEL2IDX`, `IDX2LABEL`).
- **Validazione:**
  - Split stratificato `train/valid` (default 80/20 con `train_test_split` e seed fissato).
  - Struttura pronta per estendere a `StratifiedKFold`/`GroupKFold` (basta iterare sugli split e richiamare `run_experiment`).
- **Modellazione & training:**
  - Classe `RecurrentBackbone` parametrica (tipologia cella, hidden size, layer, dropout, bidirezionalità) richiamata dal loop di training.
  - `fit_model` produce history completo (loss, accuracy, precision, recall, F1) + checkpoint del best model; supporto a TensorBoard (`logs/<run_name>`).
  - Registro esperimenti con `EXPERIMENT_CONFIGS` e tabella riassuntiva (`summary_table`) per confrontare diverse architetture.
- **Inferenza e submission:**
  - Ricarica automatica del best checkpoint, inferenza sul test set e salvataggio di `submission_<run_name>.csv` in `outputs/`.

## Come usare il notebook
1. Impostare (se serve) `EXPERIMENT_CONFIGS` con le configurazioni da provare (es. LSTM baseline, GRU bidirezionale, variazioni di dropout/hidden).
2. Eseguire la cella di training multiplo: ogni esperimento salva checkpoint, log TensorBoard e risultati in `experiment_results`.
3. Selezionare automaticamente il best run (macro-F1 più alta) e visualizzare curve di apprendimento + matrice di confusione.
4. Generare il file di submission (`submission_<run_name>.csv`) e conservare i checkpoint corrispondenti per il report.
5. Opzionale: ampliare con cross-validation, ricerca hyperparam e/o ensemble sfruttando la stessa infrastruttura.

## Punti chiave per il report finale
- Documentare configurazioni provate (parametri principali, metriche, path checkpoint) usando `summary_table` come riferimento.
- Inserire nel notebook output chiave: log TensorBoard, grafici da `plot_history`, confusion matrix e metriche sklearn.
- Allegare i notebook eseguiti, i file di submission e i checkpoint migliori coerenti con quanto descritto nel report.
