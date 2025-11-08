# Cosa facciamo ai dati dei pirati (spiegato super facile)

Immagina di avere tanti libretti di storie. Ogni libretto racconta, pagina dopo pagina, come sta un pirata: se ha male, come si muovono le sue ginocchia, quante mani ha (pirati strani!) e così via. Noi vogliamo leggere tutti questi libretti e decidere se il pirata sta bene, ha un pochino di male, oppure tanto male.

## Passo 1 — Mettiamo ordine alle pagine
- Nei file originali ogni riga è una "pagina" con un numero (`time`) e il nome del libretto (`sample_index`).
- Per allenare il computer è più comodo avere tutto il libretto per intero insieme: prendiamo tutte le pagine di uno stesso pirata e le incolliamo in ordine, così otteniamo una storia intera con 180 pagine.

## Passo 2 — Traduciamo le parole in numeri
- Alcune informazioni sono scritte come parole, per esempio se il pirata ha `two` gambe o `one` mano.
- I computer capiscono solo numeri, quindi trasformiamo `one`, `two`, `zero` in 0, 1, 2. È come fare una legenda semplice.

## Passo 3 — Puliamo il quaderno (normalizzazione)
- Alcune misure (tipo gli angoli delle ginocchia) possono avere numeri molto grandi o molto piccoli.
- Calcoliamo la media e quanto oscillano questi numeri su tutti i pirati, e poi "ridimensioniamo" tutto così nessun numero è troppo gigante o troppo minuscolo. Questo aiuta il modello a non confondersi.

## Passo 4 — Dividiamo in due gruppi
- Per capire se il modello impara davvero, separiamo alcune storie in un gruppo speciale chiamato **validazione**.
- Alleniamo il modello su un gruppo (train) e poi controlliamo se indovina anche sull'altro gruppo (valid). Così capiamo se è bravo davvero o se sta copiando.

## Perché questi passi sono importanti?
- **Ordine**: senza mettere insieme le pagine il modello leggerebbe pezzi sparsi e non capirebbe la storia completa.
- **Numeri**: senza trasformare le parole, il computer non potrebbe fare calcoli.
- **Pulizia**: senza normalizzare, alcune feature comanderebbero troppo e altre sarebbero ignorate.
- **Validazione**: senza un gruppo di controllo non sapremmo se il modello funziona sui pirati "nuovi".

## Cosa userà il modello?
- TUTTE le feature numeriche (dolore percepito, misure dei 31 giunti, quante mani/gambe/occhi) dopo essere state trasformate e normalizzate.
- Le etichette finali (`no_pain`, `low_pain`, `high_pain`) convertite in numeri, così possiamo insegnare al modello a riconoscerle.

Con questi preparativi, i nostri modelli (LSTM, GRU, RNN) possono leggere le storie dei pirati in modo ordinato e imparare a dire quanto dolore hanno.
