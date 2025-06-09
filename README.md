## Struktura repozitorija
- ```data/``` mapa za podatke
- ```predstavitev/``` mapa z datotekami za predstavitev
- ```workspace/``` mapa z datotekami razvojnega okolja (ne predstavlja dela končne oddaje)
- ```final.py``` datoteka za zagon modela za napovedovanje števila koles
- ```predstavitev.pdf``` pdf datoteka s predstavitvijo
- ```requirements.txt``` datoteka s potrebnimi knjižnicami 
- ```tcn_model.pt``` streniran tcn model

## Vzpostavitev okolja
Skripta za zagon modela je namenjena zagonu v `python 3.12`. Knjižnice, ki so potrebne za zagon namestimo z
```bash
pip install -r requirements.txt
```
Skripta `final.py` brez dodatnih argumentov zažene napoved na testni množici `data/bicikelj_test.csv` in meteo podatkih za čas testne množice `data/weather_ljubljana_test.csv`.

Argumenti:
- `--train`: streniraj model `tcn_model.pt`

Primeri:
```bash
python final.py

python final.py --train
```

Za uporabo druge testne množice je potrebno pridobiti podatke o vremenu z Meteo strežnika, kjer določimo <b>start_date=...</b> in <b>end_date=...</b>:

Primer:
```
https://archive-api.open-meteo.com/v1/archive?latitude=46.05&longitude=14.51&start_date=2025-01-01&end_date=2025-05-19&hourly=temperature_2m,precipitation,windspeed_10m,cloudcover&timezone=Europe%2FBerlin&format=csv
```
