# Refactor RNN Training — Guida Operativa (EMA, Warmup, AMP, Split riproducibile, Scheduler robusto)

Questa guida **pronta all'uso** spiega **tutte le modifiche** da applicare al tuo progetto per rendere il training più **stabile, riproducibile e performante**. Ogni sezione include snippet **plug‑and‑play**. Mantieni i nomi delle tue funzioni (`prepare_config`, `create_dataloaders`, `fit_model`, `run_experiment`) e adatta solo dove indicato.

---

## 1) Config estesa (default sicuri)
Aggiorna `prepare_config` e i default della `base_config` con queste chiavi (o unisci con le tue):
```python
base_config = {
    'run_name': None,
    'rnn_type': 'gru',          # default GRU; cambia in 'lstm' se vuoi
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': False,     # True solo per uso OFFLINE (non causale)
    'lr': 2e-3,
    'weight_decay': 1e-5,
    'epochs': 60,
    'batch_size': 64,
    'valid_size': 0.1,
    'seed': 42,

    # Early stopping & selezione best
    'patience': 8,
    'min_improvement': 5e-4,       # soglia su EMA
    'metric_for_best': 'macro_f1', # metrica target

    # Stabilità dell'ottimizzazione
    'warmup_epochs': 5,
    'scheduler': 'plateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 3,
    'scheduler_threshold': 1e-3,
    'scheduler_cooldown': 0,
    'max_grad_norm': 5.0,

    # Dataloading
    'num_workers': 4,
    'pin_memory': True,

    # EMA
    'ema_alpha': 0.1,

    # Logging
    'tensorboard': True,

    # Imbalance
    'label_smoothing': 0.05,
    'use_focal_loss': False,
    'use_class_weights': False,
}
```

Aggiungi un **seed globale** e chiamalo all’inizio di `run_experiment`:
```python
def set_seed(s=42):
    import os, random, numpy as np, torch
    random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# in run_experiment(...)
set_seed(config.get('seed', 42))
```

---

## 2) Split riproducibile + DataLoader veloce
Restituisci **anche gli indici** di train/val e usa un `generator` seed‑ato per lo shuffle deterministico.
```python
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def create_dataloaders(X, y, valid_size=0.1, batch_size=64, seed=42, num_workers=4, pin_memory=True):
    import numpy as np, torch
    idx_all = np.arange(len(y))
    X_tr, X_val, y_tr, y_val, tr_idx, val_idx = train_test_split(
        X, y, idx_all, test_size=valid_size, random_state=seed, stratify=y
    )

    tr_ds = TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).long())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers>0),
        generator=gen
    )
    valid_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers>0)
    )
    return train_loader, valid_loader, (X_tr, y_tr, X_val, y_val), (tr_idx, val_idx)
```

Aggiorna la chiamata in `run_experiment`:
```python
train_loader, valid_loader, (X_tr, y_tr, X_val, y_val), (tr_idx, val_idx) = create_dataloaders(
    X_train_np, y_train_idx,
    valid_size=config.get('valid_size', 0.1),
    batch_size=config['batch_size'],
    seed=config.get('seed', 42),
    num_workers=config.get('num_workers', 4),
    pin_memory=config.get('pin_memory', True),
)
```

---

## 3) Loss con pesi o Focal (toggle via config)
Definisci la loss in modo flessibile:
```python
import torch, torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = (1-pt)**self.gamma * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()

def build_criterion(config, class_counts=None):
    weight = None
    if config.get('use_class_weights', False) and class_counts is not None:
        total = sum(class_counts)
        weight = torch.tensor([total/c for c in class_counts], dtype=torch.float32)
    if config.get('use_focal_loss', False):
        return FocalLoss(alpha=weight, gamma=2.0)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=config.get('label_smoothing', 0.0))
```
Usa `build_criterion(...)` quando crei `criterion` (passa `class_counts` se li hai).

---

## 4) Trainer: AMP, Warmup, EMA su macro‑F1, Scheduler robusto, Early‑Stopping
Integra questi blocchi in `fit_model`:

```python
import torch
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix, f1_score

scaler = GradScaler()
alpha = config.get('ema_alpha', 0.1)
best_ema = -float('inf')
ema_val = None
patience_counter = 0

# Scheduler plateau su macro‑F1 (mode='max')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max',
    factor=config.get('scheduler_factor', 0.5),
    patience=config.get('scheduler_patience', 3),
    threshold=config.get('scheduler_threshold', 1e-3),
    cooldown=config.get('scheduler_cooldown', 0),
)

warmup_epochs = config.get('warmup_epochs', 5)
base_lr = config['lr']

for epoch in range(1, config['epochs']+1):
    # ---------- TRAIN ----------
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        scaler.step(optimizer); scaler.update()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    # ---------- VALID ----------
    model.eval()
    val_loss = 0.0
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_tgts.append(yb.cpu())
    val_loss /= len(valid_loader.dataset)
    preds = torch.cat(all_preds).numpy()
    tgts = torch.cat(all_tgts).numpy()

    macro_f1 = f1_score(tgts, preds, average='macro')
    # EMA della macro-F1
    ema_val = macro_f1 if ema_val is None else (alpha*macro_f1 + (1-alpha)*ema_val)

    # Warmup LR (prima dello step di scheduler)
    if epoch <= warmup_epochs:
        lr_now = base_lr * (epoch / warmup_epochs)
        for g in optimizer.param_groups: g['lr'] = lr_now
    else:
        scheduler.step(macro_f1)

    # Early stopping SU EMA (niente “epoca miracolosa”)
    improved = (ema_val > best_ema + config.get('min_improvement', 0.0))
    if improved:
        best_ema = ema_val
        patience_counter = 0
        best_state = {
            'model_state_dict': {k:v.cpu() for k,v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_macro_f1': macro_f1,
            'val_macro_f1_ema': ema_val,
            'val_loss': val_loss,
        }
    else:
        patience_counter += 1
        if patience_counter >= config['patience']:
            break

    # (Opzionale) TensorBoard: loss, F1 macro, EMA, LR, per‑classe e conf. matrix
    if writer is not None:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/macro', macro_f1, epoch)
        writer.add_scalar('F1/macro_ema', ema_val, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        try:
            import numpy as np, matplotlib.pyplot as plt
            cm = confusion_matrix(tgts, preds)
            rep = classification_report(tgts, preds, output_dict=True, zero_division=0)
            for cls_name, stats in rep.items():
                if isinstance(stats, dict) and 'f1-score' in stats:
                    writer.add_scalar(f'F1_class/{cls_name}', stats['f1-score'], epoch)
            fig = plt.figure()
            plt.imshow(cm, interpolation='nearest')
            plt.title('Confusion Matrix'); plt.xlabel('Pred'); plt.ylabel('True')
            for (i,j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha='center', va='center')
            writer.add_figure('ConfusionMatrix', fig, epoch)
            plt.close(fig)
        except Exception:
            pass
```

> **Nota**: mantieni il tuo `model(...)` invariato. Se usi `bidirectional=True`, assicurati che il layer finale gestisca il `hidden_size*2`.

---

## 5) Salvataggi completi in `run_experiment`
Dopo `fit_model(...)`, arricchisci `best_state` con split, config e mapping utili all’inferenza:

```python
best_state['data_split'] = {
    'X_train': X_tr, 'y_train': y_tr,
    'X_valid': X_val, 'y_valid': y_val,
}
best_state['split_idx'] = {'train_idx': tr_idx, 'valid_idx': val_idx}
best_state['config'] = copy.deepcopy(config)
best_state['run_name'] = config['run_name']

# Se presenti nel tuo progetto, aggiungi:
# best_state['feature_names'] = FEATURE_COLUMNS
# best_state['label2idx'] = LABEL2IDX
# best_state['idx2label'] = {v:k for k,v in LABEL2IDX.items()}
# best_state['scaler_mean'] = feat_mean
# best_state['scaler_std']  = feat_std
```

> **Importante**: salva sempre **solo** `state_dict` del modello (come sopra), non l’oggetto intero.

---

## 6) Scelte architetturali (GRU mono di default)
- **GRU monodirezionale** è una scelta solida per uso **causale** e dataset medi: meno parametri di LSTM, convergenza più rapida, meno overfitting.
- Valuta **Bi‑GRU** (solo offline) se vuoi spremere l’ultimo punto di F1 (a costo di causalità).
- Tieni `rnn_type` selezionabile via config (`'gru'`/`'lstm'`) per piccole ablation.

Esempi di config rapidi:
```python
# GRU mono "standard"
cfg = prepare_config('GRU_mono_256x2', {'rnn_type':'gru','bidirectional':False,'hidden_size':256,'num_layers':2,'dropout':0.3})

# Bi‑GRU (OFFLINE)
cfg = prepare_config('BiGRU_256x1', {'rnn_type':'gru','bidirectional':True,'hidden_size':256,'num_layers':1,'dropout':0.3})

# LSTM mono (check rapido)
cfg = prepare_config('LSTM_mono_256x1', {'rnn_type':'lstm','bidirectional':False,'hidden_size':256,'num_layers':1,'dropout':0.3})
```

---

## 7) Checklist rapida
- [ ] `set_seed` chiamato in `run_experiment`
- [ ] `create_dataloaders` ritorna `(tr_idx, val_idx)` e usa `generator`
- [ ] `criterion` costruita con `build_criterion` (class weights / focal / smoothing)
- [ ] `fit_model`: AMP + grad clip + **warmup** + **EMA macro‑F1** + **plateau scheduler**
- [ ] Early stopping su **EMA** (non sul picco singolo)
- [ ] `best_state`: `state_dict`, `config`, split, mapping/feature/scaler

---

## 8) Note pratiche
- Se vedi **picchi isolati** in validation: la **EMA** evita di salvare quell’epoca “miracolosa”.
- Se il **recall di high_pain** è basso: attiva `use_class_weights=True` **oppure** `use_focal_loss=True` (non entrambe).
- Se il training è lento: AMP è già incluso; puoi ridurre `epochs` e affidarti all’early stopping.

Buon lavoro! 🚀
