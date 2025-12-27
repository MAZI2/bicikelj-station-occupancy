---
title: City Bike Station Temporal Prediction (BicikeLj)
colab:
date: 2025-06-09
short: Predict bike availability at BicikeLj stations using temporal, spatial, and weather features with a TCN-based model ensemble.
long: This project focuses on temporal prediction of bike availability at BicikeLj city bike stations using historical bike counts, station embeddings, nearby station activity, holidays, and weather data...
image: https://mpog.dev/content/uozp/bicikelj/model.png
---
## City Bike Station Temporal Prediction (BicikeLj)

## Task Description
Predict the number of bikes at city bike stations over time using historical bike counts, temporal, and contextual features. The goal is to forecast bike availability for future time points at each station, leveraging both local station embeddings and nearby station usage.

---

## Model Selection

### Main Architecture
- Models evaluated individually assuming independence between nodes (training complexity).
- Relative comparison between models at the same graph level.
- Method labels:
  - Gray: untested
  - Green: selected for final model
  - Red: rejected / worse
  - Blue: comparison not necessarily relevant
  - Yellow: negligible improvement

### Features
- **Learned station embeddings**  
- **Holidays**
- **Date/Time:** Hour, Month, Weekday, Hour as cyclic features  
- **Weather:** Temperature (°C), Precipitation (mm), Windspeed (km/h), Cloud cover (%)  
- **Nearby stations:** Bike counts of 2 nearest stations  

### Models
- STGCN
- MLP
  - Shared
  - Per station
  - T-1
- LSTM
  - Shared
  - Per station
- LightGBM (Gradient boosting) per station
- Transformer
- TCN  
  - Single head
  - Multiheaded

---

## Evaluation
- Training set split: 80% train, 10% validation, 10% holdout (non-overlapping 48+4 hour windows sampled randomly).  
- Early stopping on validation MAE.  
- Hyperparameter search and model comparison done on holdout set with same seed.  
- For competition server: 90% train, 10% validation, no holdout.

### Hyperparameters

**Top 5 configurations from grid search**

| Combination | hidden_dim | dropout | lr    | weight_decay | batch_size |
|------------|------------|---------|-------|--------------|------------|
| 1          | 64         | 0.2     | 0.001 | 0.0001       | 64         |
| 2          | 64         | 0.2     | 0.0005| 0.0001       | 128        |
| 3          | 64         | 0.4     | 0.001 | 0.0001       | 128        |
| 4          | 64         | 0.4     | 0.001 | 0.0          | 64         |
| 5          | 64         | 0.2     | 0.001 | 0.0001       | 32         |

**Selected model hyperparameters**

| Batch size | LR    | Weight decay | Dropout | Hidden dim |
|------------|-------|--------------|---------|------------|
| 128        | 5e-4  | 1e-4         | 0.2     | 64         |

| Kernel size | #Blocks | Stride | Station embed dim |
|------------|---------|--------|-----------------|
| 3          | 4       | 1      | 8               |

- Early stopping patience = 8
- Other parameters like kernel_size, dilation, and number of temporal blocks selected empirically.

### Performance
- Holdout set (160 predictions across 40 48-hour sequences): MAE = **9.4291**  
- Competition server model: MAE = **9.2866**
- Training time on Nvidia A100: ~1.22s per epoch

---

## Model Explanation

- SHAP analysis highlights most important features:
  - Temporal embeddings
  - Station embeddings
  - Nearby station usage
  - Weather features contribute moderately
- Visualizations of SHAP values and feature importance included.

## Repository Structure
- `data/` folder for data
- `presentation/` folder with presentation files
- `workspace/` folder with development environment files (not part of the final submission)
- `final.py` script to run the bike count prediction model
- `presentation.pdf` PDF file with the presentation
- `requirements.txt` file with required Python libraries
- `tcn_model.pt` trained TCN model

## Environment Setup
The model script is intended to run with `python 3.12`. Install required libraries with:
```bash
pip install -r requirements.txt
```

The script `final.py` without additional arguments runs predictions on the test set `data/bicikelj_test.csv` using weather data for the test period from `data/weather_ljubljana_test.csv`.

Arguments:
- `--train`: train the model `tcn_model.pt`

Examples:

```bash
python final.py

python final.py --train
```

To use a different test set, you need to obtain weather data from the Meteo server, specifying <b>start_date=...</b> and <b>end_date=...</b>:

Example:
```
https://archive-api.open-meteo.com/v1/archive?latitude=46.05&longitude=14.51&start_date=2025-01-01&end_date=2025-05-19&hourly=temperature_2m,precipitation,windspeed_10m,cloudcover&timezone=Europe%2FBerlin&format=csv
```

