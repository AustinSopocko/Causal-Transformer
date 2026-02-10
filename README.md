# Causal Rollout Transformer (CRT)

Initial Prototype a Transformer based model for counterfactual prediction in time series data. CRT predicts future outcomes given historical states, actions, and outcomes, along with future actions ( 1 step treatments)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data_source real --data_path data/covid --model_type crt
```

### Evaluation

```bash
python evaluate.py --data_source real --data_path data/covid --model_type crt
```

## Dataset

Download COVID-19 dataset:

```bash
python download_covid_data.py
```





