# AN2DL25 Challenge 1 — Nota di Lavoro

## Obiettivo del progetto
- Classificare sequenze multivariate (180 time step, 37 feature) del dataset `Pirate Pain` in tre classi: `no_pain`, `low_pain`, `high_pain`.
- Vietato l'uso di modelli pre-addestrati: l'intera pipeline deve allenare da zero architetture ricorrenti.
- Produrre un notebook eseguibile (`challenge1_solution.ipynb`) e un file di submission `.csv` per Kaggle.

## Collegamenti con il notebook di lezione
- **Notebook di riferimento:** `old_notebooks/Timeseries Classification (1).ipynb`, orientato a Google Colab e dataset dimostrativi.
- **Principali differenze introdotte:**
  - Rimosse le istruzioni per il montaggio di Google Drive e comandi `%cd`, sostituite con path locali (`/Users/md101ta/Desktop/Pirates`).
  - Gestione esplicita del `seed`, della configurazione CUDA e della directory `outputs/` per salvare checkpoint e submission.
  - Uso di `pandas` per pivotare i dati da formato long (per time step) a tensori `(N, T, F)` compatibili con PyTorch.
  - Conversione delle pipeline di esempio basate su TensorFlow in un'implementazione PyTorch `RecurrentBackbone` parametrica (RNN/GRU/LSTM).
  - Integrazione di funzioni helper (normalizzazione, dataset/dataloader, training loop, scheduler, salvataggio miglior modello).

## Adattamento al dataset Pirate Pain
- **Pre-processing:**
  - Pivot per ottenere sequenze ordinate per `sample_index` e `time`.
  - Normalizzazione feature-wise con media e deviazione standard calcolate sull'intero training set.
  - Mappatura etichette stringa → indici (`LABEL2IDX` e `IDX2LABEL`).
- **Validazione:**
  - Split stratificato `train/valid` (80/20) mantenendo la distribuzione di classi sbilanciata.
  - Supporto a `StratifiedKFold` suggerito per esperimenti futuri.
- **Modellazione:**
  - Classe `RecurrentBackbone` configurabile (`rnn`, `gru`, `lstm`, bidirezionale, numero layer, dropout).
  - Training loop con `AdamW`, gradient clipping, scheduler `ReduceLROnPlateau` monitorando `macro F1`.
  - Log dei risultati, salvataggio `best_state` e valutazione (report sklearn + matrice di confusione).
- **Inferenza:**
  - DataLoader dedicato al test set normalizzato, generazione di `submission_lstm.csv` in `outputs/`.

## Come usare il notebook
1. Eseguire il notebook `challenge1_solution.ipynb` in locale (o su macchina con GPU, adattando i path).
2. Verificare le celle di training per un modello di base (`train_model(rnn_type='lstm', ...)`).
3. Valutare la macro-F1 in validazione e salvare la submission.
4. Esplorare varianti GRU/RNN e cross-validation per migliorare il punteggio.

## Punti chiave per il report finale
- Documentare esperimenti (ipotesi, metriche, checkpoint salvati).
- Inserire nel notebook output e grafici generati (`plot_history`, matrice di confusione).
- Conservare la corrispondenza tra configurazioni provate e file `.pt` esportati.
