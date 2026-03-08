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

Oxford panel training:

```bash
python train_oxford.py --oxford_csv data/oxford/oxford_panel.csv --config src/configs/oxford_config.yaml --model_type crt
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

Build Oxford panel dataset:

```bash
python download_oxford_panel.py
```



