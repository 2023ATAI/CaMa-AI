# CaMa-PyTorch LSTM Post-Processing

This folder contains the offline LSTM post-processing workflow for
station-level CaMa-PyTorch discharge outputs and GRDC observations.

The LSTM workflow does not modify CaMa physical states, model routing, restart
state, or the original simulation time axis.

## Preprocess

```bash
python LSTM/preprocess/00_scan_inputs.py \
  --cama-out-dir /path/to/CaMa-PyTorch/output \
  --stn-list evaluate/stn_list.txt \
  --grdc-dir /path/to/GRDC_Day \
  --out-dir ./lstm_preprocess

python LSTM/preprocess/01_build_station_qsim_qobs_us05min.py \
  --cama-out-dir /path/to/CaMa-PyTorch/output \
  --mapping-csv /path/to/station_mapping.csv \
  --grdc-dir /path/to/GRDC_Day \
  --out-dir ./lstm_preprocess
```

The build script writes the NPZ dataset used by training, typically:

```text
./lstm_preprocess/01_station_series/station_qsim_qobs_2004_2010.npz
```

## Train

```bash
python LSTM/train/train_lstm.py \
  --npz-path ./lstm_preprocess/01_station_series/station_qsim_qobs_2004_2010.npz \
  --out-dir ./lstm_training
```

The default model checkpoint is:

```text
./lstm_training/best_model.pt
```

## Evaluate

```bash
python LSTM/evaluate/evaluate_lstm.py \
  --npz-path ./lstm_preprocess/01_station_series/station_qsim_qobs_2004_2010.npz \
  --model-path ./lstm_training/best_model.pt \
  --out-dir ./lstm_evaluation
```

Training and evaluation require explicit input and output paths so the workflow
does not depend on the current working directory.
