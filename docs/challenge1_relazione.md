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
  - Implementazione PyTorch modulare (`RecurrentBackbone`) con supporto a RNN/GRU/LSTM, versioni bidirezionali/monodirezionali e hyperparam personalizzabili.
  - Training loop avanzato (`fit_model`) con mixed precision, gradient clipping, ReduceLROnPlateau monitorato su macro-F1, early stopping configurabile, checkpointing e logging TensorBoard.
  - Sistema di esperimenti (`EXPERIMENT_CONFIGS`) per lanciare sequenzialmente 6 configurazioni (LSTM/GRU/RNN × mono/bidirezionali) e confronto automatico (`summary_table`).
  - Funzioni dedicate alla cross-validation (`run_cross_validation`, `summarize_cv_results`) per replicare l’approccio di lezione con split multipli e report aggregati (media/std delle metriche).

## Adattamento al dataset Pirate Pain
- **Pre-processing:**
  - Pivot per ottenere sequenze ordinate per `sample_index` e `time` (180 step × 37 feature).
  - Normalizzazione feature-wise usando media/deviazione dello split di training.
  - Codifica di `n_legs`, `n_hands`, `n_eyes` in valori interi condivisi con il test set.
  - Mapping etichette stringa → indici (`LABEL2IDX`, `IDX2LABEL`).
- **Validazione:**
  - Split stratificato `train/valid` (default 80/20 con `train_test_split` e seed fissato) per gli esperimenti rapidi.
  - Cross-validation stratificata a 5 fold con ricostruzione dei DataLoader per ogni fold e logging dedicato.
- **Modellazione & training:**
  - Classe `RecurrentBackbone` parametrica (tipologia cella, hidden size, layer, dropout, bidirezionalità) richiamata dal loop di training.
  - Config di default più “pesante”: 256 hidden units, 3 layer, 200 epoche, patience 20, scheduler patience 5 (replica del workload mostrato a lezione).
  - `fit_model` produce history completo (loss, accuracy, precision, recall, F1) + checkpoint del best model; supporto a TensorBoard (`logs/<run_name>`).
  - Report finale degli esperimenti (`summary_table`) e, per la cross-validation, tabelle aggregate (`cv_summary`) con mean/std delle metriche.
- **Inferenza e submission:**
  - Ricarica automatica del best checkpoint, inferenza sul test set e salvataggio di `submission_<run_name>.csv` in `outputs/`.

## Come usare il notebook
1. Impostare (se serve) `EXPERIMENT_CONFIGS` con le configurazioni da provare (LSTM/GRU/RNN mono/bidirezionali o nuove varianti).
2. Eseguire la cella di training multiplo: ogni esperimento salva checkpoint, log TensorBoard e risultati in `experiment_results`.
3. Lanciare la cross-validation (`run_cross_validation`) sulla configurazione preferita per ottenere metriche medie e stabilità.
4. Consultare `summary_table`, `cv_summary` e il log TensorBoard combinato (`%tensorboard --logdir outputs/logs`) per analisi qualitative/quantitative.
5. Generare il file di submission (`submission_<run_name>.csv`) e conservare i checkpoint corrispondenti per il report.
6. Opzionale: estendere con hyperparameter search (iterando sulle configurazioni) o ensemble di checkpoint.

## Punti chiave per il report finale
- Documentare configurazioni provate (parametri principali, metriche per split, path checkpoint) usando `summary_table` e `cv_summary` come riferimento.
- Inserire nel notebook output chiave: log TensorBoard, grafici da `plot_history`, confusion matrix, tabelle CV.
- Allegare i notebook eseguiti, i file di submission e i checkpoint migliori coerenti con quanto descritto nel report.
