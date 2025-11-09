# Diario Esecuzione Notebook (stile laboratorio)

Questa guida segue l’ordine di esecuzione del notebook `challenge1_solution.ipynb`, esattamente come faresti in lab. Ogni sezione indica cosa lanciare, cosa succede e quali risultati controllare.

---

## 1. Preprocessing
**Cosa lanciare**: celle iniziali (import, seed, caricamento CSV, pivot, normalizzazione).

**Cosa succede**
- Importiamo librerie, fissiamo il seed, impostiamo le cartelle (`data`, `outputs`, `logs`).
- Leggiamo `pirate_pain_train.csv` + `pirate_pain_train_labels.csv` e il test.
- Convertiamo le colonne categoriche (`n_legs`, `n_hands`, `n_eyes`) in numeri (0, 1, 2…).
- Pivotiamo il dataset: ogni `sample_index` diventa una matrice `(180 time step × 37 feature)`.
- Normalizziamo le feature (media ≈ 0, varianza ≈ 1) e salviamo `mean/std` per l’inferenza.

**Risultato atteso**
- Vedi le shape (`(N_train, 180, 37)`), i mapping delle etichette e un grafico della distribuzione classi.
- Da qui in poi i dati sono pronti per qualsiasi modello ricorrente.

---

## 2. Allenamento RNN classiche
**Cosa lanciare**: blocco `EXPERIMENT_CONFIGS` mantenendo le entry RNN (`RNN_BI`, `RNN_SINGLE`) o eseguendo una cella dedicata se hai lasciato il set compatto.

**Cosa succede**
- Per ogni configurazione RNN (mono + bidirezionale):
  - Creiamo dataloader (split 90/10 stratificato).
  - Alleniamo per 60 epoche max (early stopping a EMA F1, warmup + scheduler).
  - Log su TensorBoard (`outputs/logs/RNN_*`).

**Cosa controllare**
- `summary_table`: due righe dedicate alle RNN con macro F1/accuracy.
- Cella `plot_history(best_history, ...)` se la RNN è la migliore: grafico loss/F1.
- Cella confusion matrix + classification report (stampa ricordata dal miglior modello corrente).

**Aspettativa**
- RNN mono di solito sotto GRU ma utile come baseline. Se la RNN ha macro F1 bassa, prosegui comunque (è il comportamento “lab”).

---

## 3. Allenamento GRU (mono e bi)
**Cosa lanciare**: nella stessa cella `EXPERIMENT_CONFIGS` (dopo aver commentato eventuali modelli che non vuoi) oppure con configurazioni dedicate (es. `prepare_config('GRU_SINGLE', {...})`, `prepare_config('GRU_BI', {...})`).

**Cosa succede**
- Set di run: GRU monodirezionale + bidirezionale.
- Stessa pipeline (loss pesata se `use_class_weights=True`, EMA, scheduler).
- Log separati: `outputs/logs/GRU_SINGLE`, `outputs/logs/GRU_BI`.

**Cosa controllare**
- `summary_table`: le righe GRU in cima (solitamente macro F1 più alta).
- Grafico `plot_history` riflette l’andamento GRU (loss/accuracy/F1/EMA).
- Confusion matrix: utile per vedere come trattano `high_pain` vs `low_pain`.
- Se vuoi approfondire: lancia `run_cross_validation` su GRU per il report finale.

**Aspettativa**
- GRU monodirezionale è spesso il best model. Prendi nota di F1, accuracy, epoca migliore.

---

## 4. Allenamento LSTM
**Cosa lanciare**: Rimuovi/commenta gli altri run e lascia le configurazioni LSTM (`LSTM_BI`, `LSTM_SINGLE`).

**Cosa succede**
- Alleni LSTM monodirezionale e bidirezionale con la stessa pipeline.
- Anche qui log separati (`outputs/logs/LSTM_*`).

**Cosa controllare**
- `summary_table`: confronto immediato con GRU e RNN.
- Grafico `plot_history` (loss/F1, eventuale overfitting).
- Confusion matrix: spesso LSTM fatica rispetto a GRU, ma serve al confronto.

**Aspettativa**
- LSTM bidirezionale potrebbe avvicinarsi alla GRU base; il monodirezionale chiarisce quanto conti la direzione.

---

## 5. Grafici e report finali
**Cosa lanciare**
- Cella `summary_table` per ordinare i run (macro F1 decrescente).
- Cella `best_run` → stampa del modello migliore + figure.
- `plot_history(best_history, ...)` per il grafico completo.
- `classification_report` + `confusion_matrix` del best run.
- Cella `submission` (salva `submission_<run_name>.csv`).

**Cosa controllare**
- Macro F1 del best run (di solito GRU_SINGLE).
- F1 per classe, in particolare `high_pain` (per capire se serve class weighting o focal loss).

---

## 6. TensorBoard (visione globale)
**Cosa lanciare**
- In Colab: `%tensorboard --logdir "/content/drive/MyDrive/Pirates/outputs/logs"`
- In locale: `tensorboard --logdir /Users/.../Pirates/outputs/logs`

**Cosa vedrai**
- Grafici sovrapposti di loss, macro F1, accuracy per tutti i run (RNN, GRU, LSTM).
- Sezione scalars: `F1_class/<nome>` per ogni classe.
- Plugin "Graphs" non è usato, ma "Scalars" e "Images" mostrano confusion matrix (quando loggata).

**Aspettativa**
- Puoi confrontare l’EMA dei vari modelli, individuare se uno continua a migliorare oltre l’early stopping, o se vale la pena cambiare dropout, hidden size, ecc.

---

## Mini-tabella riassuntiva
| Sezione | Cosa eseguo | Output principale | Cosa imparare |
| --- | --- | --- | --- |
| Preprocessing | Celle iniziali | Dati normalizzati `(N, 180, 37)` | Dati pronti per tutti i modelli |
| RNN | `EXPERIMENT_CONFIGS` con RNN | Righe RNN in `summary_table` + grafici | Baseline semplice |
| GRU | `EXPERIMENT_CONFIGS` con GRU | GRU top performer, grafici e confusion matrix | Modello migliore + spunti F1 |
| LSTM | `EXPERIMENT_CONFIGS` con LSTM | Confronto con GRU/RNN | Valutare se vale la pena tenerle |
| Report finale | `summary_table`, `best_run`, submission | F1, grafici, CSV | Riassunto da portare nel report |
| TensorBoard | `%tensorboard --logdir...` | Loss/F1 per run, matrici, F1 classi | Diagnostica completa |

---

## Nota
- Se non ti interessa un gruppo (es. RNN), commenta la riga in `EXPERIMENT_CONFIGS` e rilancia.
- Per riprodurre il flusso “lab”: lancia i blocchi in quest’ordine, guardando i grafici dopo ogni gruppo (RNN → GRU → LSTM) così capisci subito come cambia il comportamento.
- Quando trovi la configurazione finale, puoi fare cross-validation dedicata (`run_cross_validation`) o auto (`ENABLE_AUTO_CV=True`).

Così il notebook segue lo stesso percorso del laboratorio: prima pulisci i dati, poi confronti RNN, GRU, LSTM con grafici e confusion matrix, e alla fine apri TensorBoard per vedere tutto insieme. 🚀
