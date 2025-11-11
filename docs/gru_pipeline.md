# GRU Training Pipeline Overview

## 1. Dataset Preparation
- Carica `pirate_pain_train.csv`, `pirate_pain_train_labels.csv` e `pirate_pain_test.csv` da `data/` su Google Drive.
- Converte i conteggi (`n_legs`, `n_hands`, `n_eyes`) da stringhe a numeri e forza gli altri campi a tipo numerico (scartando eventuali valori anomali).
- Applica un fattore di enfasi (`FOCUS_SCALE = 2.0`) ai sensori `joint_10` e `joint_28` prima della normalizzazione per aumentare la loro influenza.

## 2. Feature Selection
- Elimina le colonne con varianza zero.
- Aggrega per `sample_index`, calcola il **contrast** tra `high_pain` e le altre classi e seleziona le Top‑K feature (tenendo sempre `time`, `joint_10`, `joint_28`, `pain_survey_1`, `pain_survey_2`).

## 3. Sequenze & Finestra
- Ordina i frame per tempo e costruisce sequenze di 160 step.
- Analizza l'autocorrelazione delle feature chiave e ricava `WINDOW_LENGTH` (taglio minimo 30 se l'ACF lo suggerisce). Le sequenze usate dalla GRU vengono troncate agli ultimi `WINDOW_LENGTH` step.
- Calcola comunque la **media completa** di `joint_10` per ogni pirata (servirà alla regola).

## 4. Normalizzazione
- Calcola media e dev. standard sul train e applica lo **z-score** a train/val/test (non vengono più riscalate le feature dopo la normalizzazione).

## 5. Dataloader & Pesi
- Crea `Dataset` PyTorch e `WeightedRandomSampler`.
- Calcola i pesi di classe, moltiplicando quello di `high_pain` per 3.
- Usa batch size 64.

## 6. Modello & Training
- GRU monodirezionale con `hidden_size=256`, `num_layers=2`, `dropout=0.3`.
- Ottimizzatore `AdamW` (`lr=5e-4`, `weight_decay=1e-4`).
- Loss `CrossEntropyLoss` con pesi di classe e `label_smoothing=0.05`; gradient clipping 1.0 ed early stopping su macro-F1 validazione.

## 7. Regola Post-Predizione
- Dopo l'inferenza calcola la media completa di `joint_10` e `joint_28` per pirata.
- Se `joint_10 mean < 0.62` **e** `joint_28 mean > 0.08`, forza la classe a `high_pain`; altrimenti lascia la previsione.
- Applica la regola sia su validation (per il report/heatmap) sia sul test prima del submission.

## 8. Output & Artefatti
- Report di classification e **confusion matrix heatmap** su validation (dopo la regola).
- Salva: `gru_monodirectional.pt`, `feature_normalisation.npz`, `label_mapping.json`, `rule_metadata.json` e il submission CSV (`gru_predictions.csv`).
