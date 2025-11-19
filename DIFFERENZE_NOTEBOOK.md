# Confronto tra `da sistemare.ipynb` e `GABRI.ipynb`

## Panoramica Generale

Questo documento descrive le principali differenze tra i due notebook per la Challenge #1 - Pirate Pain Classification. Entrambi i notebook affrontano lo stesso problema di classificazione di serie temporali multivariate, ma con approcci metodologici e architetturali significativamente diversi.

---

## 1. Struttura e Organizzazione

### `da sistemare.ipynb`
- ✅ **Struttura molto organizzata** con sezioni markdown esplicative prima di ogni cella di codice
- ✅ **Tabella dei contenuti** all'inizio del notebook
- ✅ **Documentazione completa in inglese** con spiegazioni dettagliate
- ✅ **Codice ben commentato** e professionale
- ✅ **Struttura modulare** con funzioni ben separate

### `GABRI.ipynb`
- ⚠️ **Struttura più lineare** con meno documentazione markdown
- ⚠️ **Commenti principalmente in italiano** (alcuni in inglese)
- ⚠️ **Meno sezioni esplicative** tra le celle di codice
- ✅ **Codice funzionale** ma meno documentato

---

## 2. Preprocessing dei Dati

### 2.1 Feature Engineering

#### `da sistemare.ipynb`
- **Feature removal sistematico**: Rimuove feature basate su:
  - Costanti (no variance)
  - Bassa varianza (< 2e-3)
  - Bassa magnitudo (< 1e-3)
  - Alta correlazione (> 0.98)
  - Outlier-dominated (> 75% valori oltre 5 sigma)
- **Temporal features**: Aggiunge `time_fraction`, `time_sin`, `time_cos`, `time_is_start`, `time_is_end`
- **Temporal statistics**: Calcola rolling mean e std su finestre temporali (5, 15)
- **Categorical encoding**: Mappa semplicemente le categorie a interi (n_legs, n_hands, n_eyes)
- **Normalizzazione**: Z-score normalization (mean=0, std=1)

#### `GABRI.ipynb`
- **Embedding layers per feature categoriche**: Usa `nn.Embedding` per:
  - `pain_survey_1-4` (4 embedding separati)
  - `n_legs`, `n_hands`, `n_eyes` (3 embedding separati)
- **Robust Scaling**: Usa median e IQR invece di mean e std (più robusto agli outlier)
- **Temporal features**: Solo `time_sin` e `time_cos` (encoding ciclico)
- **Nessuna rimozione sistematica di feature**: Mantiene tutte le feature originali
- **Gestione intelligente delle feature costanti**: Imposta a 0.0 invece di rimuoverle

**Differenza chiave**: GABRI usa embeddings sofisticati, mentre `da sistemare` usa encoding semplice + normalizzazione standard.

### 2.2 Gestione del Class Imbalance

#### `da sistemare.ipynb`
- ✅ **Oversampling**: Duplica campioni delle classi minoritarie (high-pain, low-pain)
- ✅ **Data augmentation aggressiva** per high-pain samples:
  - Jittering (Gaussian noise)
  - Scaling random
  - Time shifting
  - Time flipping
  - Time masking
- ✅ **Class weights**: Pesi normalizzati e scalati per penalizzare misclassificazioni delle classi minoritarie
- ✅ **Focal Loss**: Loss function che focalizza l'apprendimento su esempi difficili

#### `GABRI.ipynb`
- ❌ **Nessun oversampling**
- ❌ **Nessuna data augmentation**
- ✅ **Class weights**: Usa `compute_class_weight` di sklearn con fattore di aggiustamento `a=0.7`
- ⚠️ **CrossEntropy Loss**: Loss standard con class weights (non Focal Loss)
- ✅ **Label smoothing**: Usa smoothing factor 0.02 per regolarizzazione

**Differenza chiave**: `da sistemare` usa tecniche aggressive di bilanciamento, mentre GABRI si affida principalmente ai class weights.

---

## 3. Architettura del Modello

### 3.1 Architettura Base

#### `da sistemare.ipynb` - `RecurrentBackbone`
```python
- Optional 1D Convolutional layers (pre-processing)
- RNN layer (RNN/GRU/LSTM, configurable)
  - Bidirectional support
  - Configurable hidden size, num_layers, dropout
- Classification head (FC layers with dropout)
```

**Caratteristiche**:
- Architettura flessibile e configurabile
- Supporta RNN, GRU, LSTM
- Opzionale: layer convolutivi 1D prima dell'RNN
- Processa sequenze bidirezionalmente
- Usa l'ultimo hidden state per la classificazione

#### `GABRI.ipynb` - `RecurrentClassifierWithEmbeddings`
```python
- Multiple Embedding layers (per feature categoriche)
- Concatenation layer (combina embeddings + features numeriche)
- RNN layer (LSTM bidirezionale)
- Attention mechanism (self-attention)
- Classification head (FC layers)
```

**Caratteristiche**:
- **Embedding layers sofisticati**: 7 embedding separati per feature categoriche
- **Attention mechanism**: Self-attention per pesare i timestep importanti
- **Architettura più complessa**: Più parametri e capacità di modellazione
- **Solo LSTM**: Non supporta RNN o GRU

**Differenza chiave**: GABRI ha un'architettura più sofisticata con embeddings e attention, mentre `da sistemare` ha un'architettura più semplice ma flessibile.

### 3.2 Sliding Windows

#### `da sistemare.ipynb`
- **Training**: Usa sliding windows di dimensione 25 con stride 15
- **Evaluation**: Usa sliding windows con aggregazione (max, mean, logsumexp)
- **Window selection**: Random durante training, centrale durante validation

#### `GABRI.ipynb`
- **Training**: Usa sliding windows di dimensione 32 con stride 16
- **Evaluation**: Aggregazione a livello utente (tutte le finestre di un utente)
- **Padding intelligente**: Riempie finestre corte con l'ultimo valore (non zeri)

**Differenza chiave**: GABRI fa aggregazione a livello utente, `da sistemare` fa aggregazione a livello finestra.

---

## 4. Training e Validazione

### 4.1 Loss Functions

#### `da sistemare.ipynb`
- **Focal Loss**: Implementazione custom con:
  - Gamma parameter (default 0.75)
  - Class weights support
  - Focalizza su esempi difficili

#### `GABRI.ipynb`
- **CrossEntropy Loss**: Loss standard con:
  - Class weights
  - Label smoothing (0.02)

**Differenza chiave**: `da sistemare` usa Focal Loss (più adatto per classi sbilanciate), GABRI usa CrossEntropy standard.

### 4.2 Training Loop

#### `da sistemare.ipynb`
- **Mixed precision training**: Usa `autocast` e `GradScaler`
- **Gradient clipping**: Max grad norm = 5.0
- **Early stopping**: Basato su F1-score validation
- **Learning rate scheduling**: ReduceLROnPlateau
- **TensorBoard logging**: Completo per tutte le metriche

#### `GABRI.ipynb`
- **Mixed precision training**: Usa `autocast` e `GradScaler`
- **Gradient clipping**: Max grad norm = 1.0 (più conservativo)
- **Early stopping**: Basato su F1-score validation
- **Learning rate scheduling**: ReduceLROnPlateau
- **L1/L2 regularization**: Aggiunge penalità L1 e L2 ai pesi
- **TensorBoard logging**: Completo

**Differenza chiave**: GABRI usa regolarizzazione L1/L2 esplicita, `da sistemare` si affida solo a dropout.

### 4.3 Cross-Validation

#### `da sistemare.ipynb`
- **Stratified K-Fold**: 5-fold CV a livello sample
- **Split a livello finestra**: Le finestre dello stesso utente possono finire in fold diversi
- ⚠️ **Potenziale data leakage**: Stesso utente in train e validation

#### `GABRI.ipynb`
- **User-level K-Fold**: Split a livello utente (sample_index)
- ✅ **Nessun data leakage**: Tutte le finestre di un utente nello stesso fold
- **Metodologia corretta**: Previene leakage tra train e validation

**Differenza chiave**: GABRI usa CV a livello utente (corretto), `da sistemare` usa CV a livello finestra (potenziale leakage).

---

## 5. Inference e Ensemble

### 5.1 Inference Strategy

#### `da sistemare.ipynb`
- **Sliding window aggregation**: Estrae multiple finestre e aggrega predizioni
- **Aggregation methods**: max, mean, logsumexp
- **Single model**: Usa il miglior modello singolo

#### `GABRI.ipynb`
- **User-level aggregation**: Aggrega tutte le finestre di un utente
- **Ensemble di 4 modelli**: Combina predizioni di 4 modelli diversi
- **Double aggregation**: Prima a livello finestra, poi a livello modello

**Differenza chiave**: GABRI usa ensemble di modelli, `da sistemare` usa un singolo modello.

---

## 6. Feature Selection e Data Profiling

### `da sistemare.ipynb`
- ✅ **Comprehensive data profiling**:
  - Descriptive statistics
  - Missing values analysis
  - Variance analysis
  - Correlation analysis
  - Outlier detection
- ✅ **Feature removal log**: Traccia tutte le feature rimosse con ragioni
- ✅ **Reports salvati**: Tutti i report salvati in directory `reports/`

### `GABRI.ipynb`
- ⚠️ **Data exploration limitata**: Solo controlli base (missing, duplicates)
- ⚠️ **Nessun profiling sistematico**: Non analizza varianza, correlazioni, outlier
- ⚠️ **Nessuna rimozione di feature**: Mantiene tutte le feature

**Differenza chiave**: `da sistemare` fa un'analisi molto più approfondita dei dati.

---

## 7. Documentazione e Presentazione

### `da sistemare.ipynb`
- ✅ **Documentazione completa in inglese**
- ✅ **Celle markdown esplicative** prima di ogni sezione
- ✅ **Commenti in inglese** nel codice
- ✅ **Struttura professionale** adatta per presentazione accademica
- ✅ **Table of contents** all'inizio

### `GABRI.ipynb`
- ⚠️ **Documentazione limitata**
- ⚠️ **Commenti principalmente in italiano**
- ⚠️ **Meno celle markdown esplicative**
- ⚠️ **Struttura più lineare** e meno organizzata

**Differenza chiave**: `da sistemare` è molto più presentabile e professionale.

---

## 8. Configurazione e Riproducibilità

### `da sistemare.ipynb`
- ✅ **SEED = 2024**
- ✅ **Deterministic training**: `cudnn.deterministic = True`, `benchmark = False`
- ✅ **Path detection automatica**: Rileva automaticamente il path corretto
- ✅ **Supporto Colab e locale**: Gestisce entrambi gli ambienti

### `GABRI.ipynb`
- ✅ **SEED = 42**
- ⚠️ **Benchmark mode**: `cudnn.benchmark = True` (più veloce ma meno riproducibile)
- ⚠️ **Path hardcoded**: Path specifico per Colab
- ⚠️ **Solo Colab**: Non gestisce ambiente locale

**Differenza chiave**: `da sistemare` è più riproducibile e portabile.

---

## 9. Risultati e Performance

### Metriche Chiave

| Aspetto | `da sistemare.ipynb` | `GABRI.ipynb` |
|---------|----------------------|---------------|
| **Best F1 Score** | ~0.985 (GRU_BI_fold1) | Da verificare |
| **Architettura** | Semplice ma flessibile | Complessa con embeddings |
| **Bilanciamento** | Aggressivo (oversampling + augmentation) | Conservativo (solo weights) |
| **Loss Function** | Focal Loss | CrossEntropy + weights |
| **CV Strategy** | Sample-level (potenziale leakage) | User-level (corretto) |
| **Ensemble** | No | Sì (4 modelli) |
| **Feature Selection** | Sistematico | Nessuno |
| **Documentazione** | Eccellente | Limitata |

---

## 10. Punti di Forza e Debolezze

### `da sistemare.ipynb`

**Punti di Forza**:
- ✅ Documentazione eccellente e professionale
- ✅ Tecniche aggressive di bilanciamento (oversampling + augmentation)
- ✅ Focal Loss per classi sbilanciate
- ✅ Feature selection sistematico
- ✅ Data profiling completo
- ✅ Codice modulare e ben organizzato
- ✅ Supporto multi-ambiente (Colab + locale)

**Debolezze**:
- ⚠️ CV a livello finestra (potenziale data leakage)
- ⚠️ Architettura più semplice (no embeddings, no attention)
- ⚠️ Nessun ensemble
- ⚠️ Normalizzazione standard (meno robusta agli outlier)

### `GABRI.ipynb`

**Punti di Forza**:
- ✅ Architettura sofisticata (embeddings + attention)
- ✅ CV a livello utente (metodologia corretta)
- ✅ Ensemble di modelli
- ✅ Robust Scaling (più robusto agli outlier)
- ✅ L1/L2 regularization esplicita
- ✅ Label smoothing

**Debolezze**:
- ⚠️ Nessun oversampling o augmentation
- ⚠️ CrossEntropy standard (meno adatto per classi sbilanciate)
- ⚠️ Documentazione limitata
- ⚠️ Nessuna feature selection
- ⚠️ Meno riproducibile (benchmark mode)
- ⚠️ Solo ambiente Colab

---

## 11. Raccomandazioni per Miglioramenti

### Per `da sistemare.ipynb`:
1. **Implementare CV a livello utente** per evitare data leakage
2. **Aggiungere ensemble** di modelli per migliorare robustezza
3. **Considerare Robust Scaling** invece di Z-score per outlier
4. **Aggiungere attention mechanism** per migliorare capacità di modellazione

### Per `GABRI.ipynb`:
1. **Aggiungere oversampling e augmentation** per bilanciare classi
2. **Implementare Focal Loss** invece di CrossEntropy
3. **Migliorare documentazione** con celle markdown esplicative
4. **Aggiungere feature selection sistematico**
5. **Rendere più riproducibile** (deterministic mode)
6. **Supportare ambiente locale** oltre a Colab

---

## 12. Conclusioni

I due notebook rappresentano approcci complementari:

- **`da sistemare.ipynb`**: Eccelle in **documentazione, bilanciamento delle classi, e organizzazione del codice**. È più adatto per presentazioni accademiche e per apprendere le tecniche.

- **`GABRI.ipynb`**: Eccelle in **architettura sofisticata, metodologia corretta (CV a livello utente), e ensemble**. È più adatto per ottenere performance competitive.

**Approccio ideale**: Combinare i punti di forza di entrambi:
- Architettura sofisticata di GABRI (embeddings + attention)
- Tecniche di bilanciamento di `da sistemare` (oversampling + augmentation)
- CV a livello utente di GABRI
- Documentazione e organizzazione di `da sistemare`
- Ensemble di modelli di GABRI
- Feature selection di `da sistemare`

