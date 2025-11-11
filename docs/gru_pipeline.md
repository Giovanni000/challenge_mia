# GRU Training Pipeline Overview

## 1. Dataset Preparation
- Carica `pirate_pain_train.csv`, `pirate_pain_train_labels.csv` e `pirate_pain_test.csv` da `data/` su Google Drive.
- Converte i conteggi (`n_legs`, `n_hands`, `n_eyes`) da stringhe (“two”) a numeri e forza tutti gli altri campi a tipo numerico.
- Applica un fattore di enfasi (`FOCUS_SCALE = 2.0`) ai sensori `joint_10` e `joint_28` per renderli più influenti nelle fasi successive.

## 2. Feature Selection
- Elimina le colonne a varianza zero.
- Aggrega per `sample_index` e calcola il **contrast** tra `high_pain` e le altre classi.
- Seleziona le Top-K feature più discriminanti mantenendo sempre `time`, `joint_10`, `joint_28`, `pain_survey_1`, `pain_survey_2`.

## 3. Sequenze & Finestra
- Ordina i frame per tempo, costruisce sequenze di lunghezza 160 e calcola la finestra usando l'autocorrelazione (minimo 60 step).
- Trunca ogni sequenza agli ultimi `WINDOW_LENGTH` elementi per l'allenamento, ma calcola la **media globale di `joint_10`** su tutta la finestra originale per la regola successiva.

## 4. Normalizzazione ed Enfasi
- Calcola media e deviazione standard sul train, applica lo **z-score** a train/val/test.
- Dopo la normalizzazione, riapplica il fattore di enfasi sui due joint chiave per amplificare i gradienti a loro associati.

## 5. Dataloader & Pesi
- Costruisce `Dataset` PyTorch e uno `WeightedRandomSampler`.
- Calcola i pesi di classe e moltiplica quello di `high_pain` per 10 per contrastare l'imbalance; batch size fissato a 128.

## 6. Modello & Training
- GRU monodirezionale con `hidden_size=320`, `num_layers=3`, `dropout=0.35`.
- Ottimizzatore `AdamW` con `lr=2e-3`, weight decay `1e-4`.
- Loss `CrossEntropyLoss` con pesi di classe e `label_smoothing=0.05`; gradient clipping a 1.0 e early stopping sul macro-F1 di validation.

## 7. Regola Post-Predizione
- Dopo l'inferenza su validation e test calcola la media di `joint_10` per pirata (sulla sequenza completa).
- Se la media scende sotto 0.64, forza la classe a `high_pain`; altrimenti mantiene la previsione del modello.
- Produce report, confusion matrix (anche in heatmap) e salva il submission CSV basato sulle predizioni corrette.

## 8. Artefatti Salvati
- Pesi della rete (`gru_monodirectional.pt`), scaler (`feature_normalisation.npz`), mapping etichette (`label_mapping.json`) e eventuale `gru_predictions.csv`.
