# Cosa fa ogni blocco del notebook (spiegato facilissimo)

Immagina che il notebook sia un grande ricettario: ogni blocco è un ingrediente o un passaggio per preparare il "modello pirata". Ecco cosa succede, passo dopo passo.

---

## 1. Prepariamo la cucina
- **Import**: prendiamo gli attrezzi (librerie Python) che useremo dopo.
- **Impostiamo il seme**: `set_seed(42)` per far sì che ogni volta il risultato sia uguale (niente sorprese!).
- **Capire dove siamo**: stampiamo se siamo su Google Colab o sul computer locale e fissiamo le cartelle (`data`, `outputs`).
- **Helper AMP**: `autocast_context()` e `create_grad_scaler()` ci permettono di usare la GPU con gli steroidi (mixed precision) senza errori.

_Cosa ci aspettiamo_: avere tutto il necessario pronto, senza doverci ricordare mille comandi a mano.

---

## 2. Leggiamo i dati e li mettiamo in ordine
- Carichiamo i CSV (`pirate_pain_train.csv`, `pirate_pain_train_labels.csv`, `pirate_pain_test.csv`).
- Convertiamo le parole (`one`, `two`) in numeri per `n_legs`, `n_hands`, `n_eyes`.
- Guardiamo quante volte compare ogni etichetta (bar chart).
- Facciamo un "pivot" per trasformare le tabelle in un formato adatto alla rete neurale: da righe sparse a blocchi `(campioni, 180, 37)`.
- Normalizziamo: tutte le caratteristiche hanno media ~0 e varianza ~1.

_Cosa ci aspettiamo_: dati puliti e pronti, così il modello non si confonde con formati strani.

---

## 3. Dataset e DataLoader
- `TimeSeriesDataset` converte gli array in oggetti PyTorch.
- `make_dataloader_from_arrays` crea carrelli (batch) con shuffle deterministico.
- `create_dataloaders` divide in train/valid (80/20 di default), restituisce anche gli indici e usa un seed fisso.

_Cosa ci aspettiamo_: ogni volta che alleniamo, gli stessi campioni finiscono nello stesso split; i carrelli sono efficienti e si riempiono senza intoppi.

---

## 4. Configurazione base
- `prepare_config` imposta i parametri standard (GRU mono, hidden 256, 2 layer, dropout 0.3, lr 2e-3… con opzioni per warmup, EMA, scheduler).

_Cosa ci aspettiamo_: cambiare esperimenti sovrascrivendo solo quello che serve (`hidden_size`, `dropout`, ecc.).

---

## 5. Perdita intelligente
- `FocalLoss` e `build_criterion` ci permettono di usare pesi di classe, focal loss, label smoothing.
- Lo decidiamo dalla configurazione (`use_class_weights`, `use_focal_loss`).

_Cosa ci aspettiamo_: se una classe (es. `high_pain`) è rara, il modello riceve più attenzione.

---

## 6. Il cuore del modello
- `RecurrentBackbone` è il cervello: GRU/LSTM/RNN con il numero di layer, hidden e dropout che scegliamo.

_Cosa ci aspettiamo_: gestisce qualsiasi variante senza cambiare altro codice.

---

## 7. Funzioni di allenamento
- `compute_classification_metrics` calcola accuracy, precision, recall, macro F1.
- `train_one_epoch` e `evaluate_epoch` gestiscono avanti/indietro sulla GPU con AMP, grad-clip e restituiscono metriche.

_Cosa ci aspettiamo_: una cornice stabile per ogni epoch, con i numeri pronti per il log.

---

## 8. Trainer super completo (`fit_model`)
- Applica warmup del learning rate, ReduceLROnPlateau, EMA sulla macro F1, early stopping sulla EMA, gradient clipping.
- Logga su TensorBoard (loss, F1, confusion matrix).
- Salva il best checkpoint (state_dict CPU-safe) con config, metriche e storia.

_Cosa ci aspettiamo_: il training si ferma da solo quando smette di migliorare, e conserva tutte le info per riprodurre i risultati.

---

## 9. Un esperimento completo (`run_experiment`)
- `set_seed` per riproducibilità.
- Crea dataloader con seed, calcola `class_counts` e la loss (pesata se serve).
- Chiama `fit_model` e salva nel risultato: modello, split, scaler, label mapping, history.

_Cosa ci aspettiamo_: ogni run è un “pacchetto completo” già pronto per inferenza o report.

---

## 10. Cross-Validation (`run_cross_validation`)
- StratifiedKFold su `n_splits` (di default 5), seed diverso per ogni fold.
- Per ogni fold allena, salva tutto come in `run_experiment` e indica il numero del fold.

_Cosa ci aspettiamo_: media e varianza delle metriche, per capire se il modello è stabile tra split diversi.

---

## 11. Tabelle riassuntive
- `build_summary_table` raccoglie i risultati da `experiment_results`, `sweep_results`, `cv_results`, `auto_cv_results` e li ordina per macro F1.

_Cosa ci aspettiamo_: una vista unica per decidere qual è il modello migliore e da rieseguire.

---

## 12. Training multipli
- `EXPERIMENT_CONFIGS`: lancia le configurazioni principali (puoi tenerne solo una o molte).
- `run_cross_validation`: cv manuale.
- `SWEEP_PARAM_SPACE`: prova combinazioni a caso di hyperparam su GRU.
- `ENABLE_AUTO_CV`: se lo metti a `True`, avvia automaticamente CV sui run migliori della tabella.

_Cosa ci aspettiamo_: zero click manuali per ripetere i test; basta eseguire la cella e aspettare.

---

## 13. Report finale
- `summary_table`: stampa run, tipo modello, se è bidirezionale, F1 e path del log.
- `build_summary_table` + `best_run`: carica il modello migliore, disegna le curve (`plot_history`), stampa report sklearn, confusion matrix e salva la submission (CSV) con il nome del run.

_Cosa ci aspettiamo_: un unico post-it con risultati, grafici e la submission già pronta da caricare su Kaggle.

---

## 14. Next steps (idee extra)
- Prova Jolly: label smoothing, focal loss, class weights, hidden size più grandi, dropout diversi, ensemble, attention.

_Cosa ci aspettiamo_: sappiamo dove mettere le mani per migliorare ancora, senza riscrivere tutto.

---

## In breve
| Blocco | Cosa fa | Cosa otteniamo |
| --- | --- | --- |
| Setup | Importa librerie, setta semi | ambiente pronto & riproducibile |
| Dati | Pulisce e normalizza | tensori `(N, 180, 37)` coerenti |
| Loader | Crea batch seed-ati | input affidabili all’allenamento |
| Config | Parametri di default | alterna esperimenti velocemente |
| Loss | Class weights / Focal | bilanciamento classi rare |
| Modello | GRU/LSTM/RNN | cervello personalizzabile |
| Trainer | Warmup, EMA, scheduler | training stabile e tracciato |
| Experiments | Run multipli, CV, sweep | esplorazione automatica |
| Summary | Tabella, grafici, submission | risultati pronti per report |

Ora sai perché ogni blocco è lì e perché conviene eseguirlo in ordine. Buon viaggio con i pirati del dolore! 🏴‍☠️🥇
