# GrasslandGrowthLab v5 – Darstellung

Dieses Update enthält nur `app.py`. Die Modellskripte und Meteodaten bleiben unverändert.

Änderungen:

- Separate Grafik der Grünlandtemperatursumme entfernt.
- GTS-Wachstumsstart als gestrichelte vertikale Linie in LAI und Biomasse beibehalten.
- Täglicher Biomassezuwachs `dBM` als einklappbare Grafik unter den Hauptplots.
- Referenz und Experiment erscheinen beide im dBM-Plot, sofern der Modus beide simuliert.

`dBM` ist in der aktuellen Modellimplementierung der auf >= 0 begrenzte
Netto-Biomassezuwachs. An Schnitttagen ist `dBM = 0`. Das ist nicht
identisch mit dem täglich modellierten Wachstum `GRO`.

Installation: `app.py` im Projektverzeichnis ersetzen.
Danach Streamlit neu laden oder neu starten.
