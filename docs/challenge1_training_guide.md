# Guida Esecuzione Training — Pirate Pain Challenge

Questa guida spiega come usare il notebook `challenge1_solution.ipynb` per allenare i modelli, quali blocchi eseguire e dove modificare le impostazioni principali.

## Struttura generale del notebook
1. **Setup & dati**: import librerie, definizione path (locale/Colab), caricamento CSV, pivot, normalizzazione.
2. **Dataset/Model utilities**: classi `TimeSeriesDataset`, `RecurrentBackbone`, funzioni di training (`fit_model`, `train_one_epoch`, ecc.).
3. **Blocchi di esperimenti**:
   - `EXPERIMENT_CONFIGS`: 6 configurazioni base (LSTM/GRU/RNN × mono/bidirezionali).
   - `SWEEP_PARAM_SPACE`: sweep automatizzato di hyperparametri per GRU.
   - `run_cross_validation(...)`: cross-validation esplicita.
   - `ENABLE_AUTO_CV`: cross-validation automatizzata sui top run.
4. **Sintesi risultati**: costruzione `summary_table`, scelta best model, grafici e submission finale.

## Configurazioni principali da modificare
### 1. Training base
```python
EXPERIMENT_CONFIGS = [
    prepare_config('LSTM_BI', {'bidirectional': True}),
    prepare_config('LSTM_SINGLE', {'bidirectional': False}),
    ...
]
```
- Ogni voce richiama `prepare_config`, che definisce hyperparam di default (hidden size 256, 3 layer, 200 epoche, ecc.).
- Modifica o aggiungi entry per provare nuove configurazioni (es. dropout diverso, layer extra, scheduler differente).
- Riduci l’elenco se vuoi meno run.

### 2. Hyperparam sweep (GRU)
```python
SWEEP_PARAM_SPACE = {
    'hidden_size': [192, 256, 320],
    'num_layers': [2, 3],
    'dropout': [0.30, 0.45],
    'bidirectional': [True, False],
    'lr': [1e-3, 2e-3],
}
MAX_SWEEP_RUNS = 12
```
- `SWEEP_PARAM_SPACE`: definiisci quali valori esplorare per hidden size, layer, dropout, bidirezionalità, learning rate.
- `MAX_SWEEP_RUNS`: limita il numero di run random estratti (utile se la combinazione totale è troppa).
- Per disattivare lo sweep basta commentare/skipparne la cella oppure impostare `MAX_SWEEP_RUNS = 0`.

### 3. Cross-Validation manuale
```python
BEST_FOR_CV = prepare_config('LSTM_BI_CV', {'bidirectional': True})
cv_results = run_cross_validation(BEST_FOR_CV, n_splits=5)
```
- Scegli la configurazione da validare (es. GRU monodirezionale).
- `n_splits=5` definisce il numero di fold. Cambia se vuoi più/meno fold.
- Ogni fold salvano checkpoint, log e metriche.

### 4. Cross-Validation automatica
```python
ENABLE_AUTO_CV = False
AUTO_CV_TOP_K = 2
AUTO_CV_SPLITS = 5
```
- Setta `ENABLE_AUTO_CV = True` per lanciare automaticamente la CV sui primi `AUTO_CV_TOP_K` run della `summary_table`.
- `AUTO_CV_SPLITS` = numero di fold per ciascun run selezionato.

### 5. TensorBoard
- Ogni run scrive log in `outputs/logs/<run_name>`.
- In Colab: `%tensorboard --logdir "/content/drive/MyDrive/Pirates/outputs/logs"`
- In locale: `tensorboard --logdir /Users/.../Pirates/outputs/logs`

## Ordine di esecuzione consigliato
1. **Carica dati e definisci config** (prime celle del notebook).
2. **Esegui i 6 run base** (`EXPERIMENT_CONFIGS`).
3. (Opzionale) **Esegui lo sweep** GRU (`SWEEP_PARAM_SPACE`).
4. (Opzionale) **Esegui CV manuale** su un modello di interesse (`run_cross_validation`).
5. (Opzionale) **Attiva auto-CV** (`ENABLE_AUTO_CV = True`) e rilancia la cella corrispondente.
6. **Rigenera la `summary_table`** per avere tutti i risultati ordinati.
7. **Seleziona best model** (già fatto dalla cella successiva) e verifica `plot_history`, `classification_report`, `confusion_matrix`.
8. **Genera submission** (cella finale) e salva i checkpoint.

## Suggerimenti pratici
- **Riduci i run** se lavori senza GPU potente (es. 6 run base + 4 del sweep). Modifica `MAX_SWEEP_RUNS` o temporaneamente svuota `EXPERIMENT_CONFIGS`.
- **Hyperparam tuning mirato**: modifica manualmente `prepare_config` per testare dropout più bassi/alti, hidden size 512, learning rate 5e-4, ecc.
- **Regularizzazione avanzata**: inserisci label smoothing, mixup, SWA direttamente nelle funzioni `train_one_epoch` o `fit_model` se vuoi replicare esperimenti diversi.
- **Breakpoint**: se vuoi fermare dopo i 6 run base, commenta le celle successive (sweep, CV) finché non ti servono.

## Troubleshooting
- **TensorBoard vuoto**: verifica di aver completato almeno un training dopo aver lanciato `%tensorboard` e che i log siano nel path corretto.
- **Warning AMP**: se vuoi eliminarli cambia import in `from torch.amp import GradScaler, autocast` e aggiorna le chiamate.
- **Training troppo lungo**: riduci `epochs`, `patience`, oppure il numero di run (`MAX_SWEEP_RUNS`).
- **Metriche basse**: controlla la `summary_table` per individuare run con F1 migliori; aumenta hidden size, aggiungi bidirezionalità o adotta nuove regularizzazioni.

Con questa guida puoi modulare facilmente la complessità dell’allenamento adattandola alle tue esigenze (rapido per test, completo per riprodurre il lavoro svolto in aula).
