# TODO:
✅ Best config for LIDL BEŽIGRAD: {'hidden_dim': 32, 'dropout': 0.25, 'learning_rate': 0.001, 'weight_decay': 1e-05, 'batch_size': 32}

'hidden_dim': 32, 'dropout': 0.20, 'learning_rate': 0.001, 'weight_decay': 1e-05, 'batch_size': 32

LSTM
'hidden_dim': 32, 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 1e-05, 'batch_size': 64

LSTM shared
hidden_dim=128, dropout=0.1, lr=0.001, weight_decay=1e-05

Better shuffled ker je bolje generaliziral ...
Na ne shuffled je blo ful tezje najdt hiperparametre k bi za cel dataset delovali ... ker overfitting.
It just let it run for longer since it was not as unpredictable

HISTORY_LEN = 48
PRED_HORIZON = 4
K_NEIGHBORS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 8

HIDDEN_DIM = 64
N_LAYERS = 4
LR = 0.0005
WEIGHT_DECAY = 0.0001
DROPOUT = 0.2

EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 128


HISTORY_LEN = 48
PRED_HORIZON = 4
K_NEIGHBORS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 8
HIDDEN_DIM = 64
N_LAYERS = 4
LR = 0.0005
WEIGHT_DECAY = 0.0001
DROPOUT = 0.2
EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 128


Train:
https://archive-api.open-meteo.com/v1/archive?latitude=46.05&longitude=14.51&start_date=2022-09-01&end_date=2024-12-31&hourly=temperature_2m,precipitation,windspeed_10m,cloudcover&timezone=Europe%2FBerlin&format=csv

Test:
https://archive-api.open-meteo.com/v1/archive?latitude=46.05&longitude=14.51&start_date=2025-01-01&end_date=2025-05-19&hourly=temperature_2m,precipitation,windspeed_10m,cloudcover&timezone=Europe%2FBerlin&format=csv


Final holdout evaluation
shap better evaluation
data graphs
How does the current model work?
stride => step

-------- it scales! ----
Train samples: 4312 | Val samples: 79
Train samples: 362208 | Val samples: 6636
⏳ Running grid search over 1 combinations...

🔍 Combo 1: hidden_dim=64, dropout=0.2, lr=0.0005, weight_decay=0.0001
Holdout MSE (real units): 9.575816242372822

📊 Top 5 Results:
   hidden_dim  dropout      lr  weight_decay  val_loss  holdout_loss
0          64      0.2  0.0005        0.0001  0.279257      9.575816


hidden_dim=64, dropout=0.4, lr=0.001, weight_decay=0.0?
Train samples: 4312 | Val samples: 79
Train samples: 362208 | Val samples: 6636
⏳ Running grid search over 1 combinations...

🔍 Combo 1: hidden_dim=64, dropout=0.4, lr=0.001, weight_decay=0.0
Holdout MSE (real units): 9.195802937395635

📊 Top 5 Results:
   hidden_dim  dropout     lr  weight_decay  val_loss  holdout_loss
0          64      0.4  0.001           0.0  0.284409      9.195803
vs 256 vs prev on less reduction
Run grid search on a 0.1 reduction and see if it is the same
-----------


I have a task to predict the number of bicycles for 4x 1 hour after a 48 hour sequence. The training data is 2 year worth of such sequences. Each station also has metadata

bicikelj_metadata.csv
name,latitude,longitude
LIDL BEŽIGRAD,46.063797,14.506854
ŠMARTINSKI PARK,46.065206,14.529911
SAVSKO NASELJE 1-ŠMARTINSKA CESTA,46.062475,14.524321
ČRNUČE,46.102446,14.530213
VILHARJEVA CESTA,46.06005,14.51302
MASARYKOVA DDC,46.05763,14.514264
POGAČARJEV TRG-TRŽNICA,46.051093,14.507186
CANKARJEVA UL.-NAMA,46.052431,14.503257
ANTONOV TRG,46.041753,14.477016
PRUŠNIKOVA,46.090608,14.471637
...

bicikelj_train.csv
timestamp,LIDL BEŽIGRAD,ŠMARTINSKI PARK,SAVSKO NASELJE 1-ŠMARTINSKA CESTA,ČRNUČE,VILHARJEVA CESTA,MASARYKOVA DDC,POGAČARJEV TRG-TRŽNICA,CANKARJEVA UL.-NAMA,ANTONOV TRG,PRUŠNIKOVA,TEHNOLOŠKI PARK,KOSEŠKI BAJER,TIVOLI,TRŽNICA MOSTE,GRUDNOVO NABREŽJE-KARLOVŠKA C.,LIDL-LITIJSKA CESTA,ŠPORTNI CENTER STOŽICE,ŠPICA,ROŠKA - STRELIŠKA,BAVARSKI DVOR,STARA CERKEV,SITULA,ILIRSKA ULICA,LIDL - RUDNIK,KOPALIŠČE KOLEZIJA,POVŠETOVA - KAJUHOVA,DUNAJSKA C.-PS MERCATOR,CITYPARK,KOPRSKA ULICA,LIDL - VOJKOVA CESTA,POLJANSKA-POTOČNIKOVA,POVŠETOVA-GRABLOVIČEVA,PARK NAVJE-ŽELEZNA CESTA,ZALOG,CESTA NA ROŽNIK,HOFER-KAJUHOVA,DUNAJSKA C.-PS PETROL,STUDENEC,PARKIRIŠČE NUK 2-FF,BRATOVŠEVA PLOŠČAD,KONGRESNI TRG-ŠUBIČEVA ULICA,BS4-STOŽICE,GERBIČEVA - ŠPORTNI PARK SVOBODA,ŽIVALSKI VRT,VOKA - SLOVENČEVA,BTC CITY/DVORANA A,TRNOVO,P+R BARJE,ROŽNA DOLINA-ŠKRABČEVA UL.,KINO ŠIŠKA,BRODARJEV TRG,ZALOŠKA C.-GRABLOVIČEVA C.,DOLENJSKA C. - STRELIŠČE,ŠTEPANJSKO NASELJE 1-JAKČEVA ULICA,SOSESKA NOVO BRDO,TRŽNICA KOSEZE,ALEJA - CELOVŠKA CESTA,MERCATOR CENTER ŠIŠKA,GH ŠENTPETER-NJEGOŠEVA C.,HOFER - POLJE,VIŠKO POLJE,BONIFACIJA,P + R DOLGI MOST,DRAVLJE,POLJE,SUPERNOVA LJUBLJANA - RUDNIK,SREDNJA FRIZERSKA ŠOLA,TRG OF-KOLODVORSKA UL.,TRG MDB,TRŽAŠKA C.-ILIRIJA,PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE,MERCATOR MARKET - CELOVŠKA C. 163,SAVSKO NASELJE 2-LINHARTOVA CESTA,BREG,BTC CITY ATLANTIS,IKEA,MIKLOŠIČEV PARK,BARJANSKA C.-CENTER STAREJŠIH TRNOVO,LEK - VEROVŠKOVA,AMBROŽEV TRG,VOJKOVA - GASILSKA BRIGADA,RAKOVNIK,PREGLOV TRG,PLEČNIKOV STADION
2025-01-01T00:00Z,4.0,20.0,5.0,7.0,17.0,18.0,15.0,13.0,12.0,7.0,4.0,5.0,11.0,13.0,16.0,7.0,9.0,11.0,5.0,19.0,2.0,3.0,8.0,8.0,10.0,13.0,4.0,14.0,4.0,19.0,7.0,7.0,7.0,2.0,7.0,8.0,3.0,11.0,13.0,9.0,19.0,8.0,20.0,4.0,4.0,1.0,13.0,13.0,7.0,8.0,10.0,9.0,3.0,16.0,6.0,12.0,8.0,6.0,7.0,13.0,13.0,15.0,4.0,7.0,14.0,10.0,8.0,21.0,0.0,4.0,20.0,9.0,3.0,19.0,7.0,16.0,13.0,14.0,4.0,18.0,7.0,17.0,6.0,8.0
2025-01-01T01:00Z,5.0,20.0,5.0,7.0,15.0,18.0,15.0,7.0,11.0,7.0,4.0,8.0,5.0,11.0,11.0,6.0,10.0,13.0,5.0,14.0,3.0,2.0,10
...

the test set is the same format as train just that 4 hours are missing after 48 hour sequences. These i need to predict. For this task i can use at most pytorch and its dependencies and scikit learn. I had by far the best result with per station lightgbm, however i cannot use it as it is external library. The second best, but still not that good is lstm, which was better as shared model rather than per station. MLP instead of lstm was not better. I also learned that stations are very weakly correlated regarding spatialy, however if the range is <1km they indeed are quite correlated. What else can i explore? What is the next promising thing to try? The following is the best model with best features

