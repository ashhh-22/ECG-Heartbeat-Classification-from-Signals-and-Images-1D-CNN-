# ECG Heartbeat Classification from Signals and Images (1D-CNN)

A deep-learning pipeline that classifies single ECG heartbeats as **Normal / Abnormal** (PTB Diagnostic DB)
or into **5 beat types** (MIT-BIH), and can also read a **photo/scan of an ECG**, turn it back into
heartbeats, and classify them. Built with TensorFlow/Keras.

> **Research prototype. Not a medical device and not for diagnosis.**

## Highlights
- Residual **1D CNN** (about 180k parameters), trained with class weights and early stopping on **balanced accuracy**.
- **Image-to-beat digitiser**: finds lead strips, reads the paper grid to get real time scale, follows the trace, cuts beats like the training data.
- **Image round-trip training**: training beats are drawn as images (different grids, resolutions, JPEG quality), read back with the real digitiser, and added to training.
- EDA, cleaning, evaluation (confusion matrix, ROC, validation-selected threshold) and CSV/image prediction scripts.

## Dataset
Kaggle **ECG Heartbeat Categorization Dataset**, built from two PhysioNet databases. It is **not included** in this repo (the MIT-BIH train file is about 400 MB); download it and put the four CSV files in a `data/` folder.

| File | Source | Rows | Task |
|---|---|---|---|
| `mitbih_train.csv`, `mitbih_test.csv` | MIT-BIH Arrhythmia DB | 87,554 / 21,892 | 5 classes: N, S, V, F, Q |
| `ptbdb_normal.csv`, `ptbdb_abnormal.csv` | PTB Diagnostic ECG DB | 4,046 / 10,506 | Normal vs Abnormal |

Each row has 188 columns: 187 signal samples (125 Hz, scaled 0-1, zero-padded) and the class label. A row starts at an R-peak.
(Note: these are PTBDB files, not PTB-XL.)

## Project structure
```
config.py        paths, seed, epochs, INPUT_LEN
data.py          loading, validation, splits
eda.py           dataset summary and plots
preprocess.py    cleaning (duplicates, flat rows, out-of-range)
model.py         1D residual CNN
augment.py       training augmentation (pen width, noise, time scale)
synth.py         renders beats as images and reads them back (round-trip data)
train.py         training + evaluation
evaluate.py      metrics, plots, threshold selection
digitize.py      ECG image -> heartbeats (grid, strips, trace tracking, beat cutting)
predict.py       predict from an ECG image
predict_csv.py   predict from a CSV of beats
trained/         pre-trained PTBDB model
samples/         small test CSVs (30 unseen beats, with and without labels)
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
Open `config.py` and set `DATASET_DIR` to the folder holding the four CSV files.

## Usage
```bash
python eda.py                              # dataset summary + plots -> Outputs_v2/eda
python preprocess.py                       # cleaned arrays + report
python train.py ptb                        # train Normal/Abnormal model (python train.py mit for 5 classes)
python predict_csv.py samples/sample_beats_labeled.csv
python predict.py path/to/ecg_image.png    # prediction from an image
```
To skip training, copy `trained/ptb_model.keras` into your output folder (`Outputs_v2`).
For a 12-lead page, set `CROP = (left, top, right, bottom)` in `predict.py` to isolate one lead.

## Method
1. **Input:** the first 80 samples of each beat (`INPUT_LEN = 80`). Using the full row let the model read how dataset rows end (beat length / tail differ by class), a shortcut that cannot be reproduced from images.
2. **Model:** conv stem, three residual blocks (32/64/128), average + max pooling, dropout, softmax. BatchNorm momentum 0.9.
3. **Training:** stratified 70/15/15 split (PTBDB), balanced class weights, Adam, early stopping on validation balanced accuracy.
4. **Threshold:** the abnormal decision threshold is chosen on validation data only (Youden's J) and applied once to the test set.
5. **Images:** grid-based time calibration (25 mm/s), colour-grid removal, trace following, R-peak to R-peak beat cutting resampled to 125 Hz.

## Results (PTBDB, held-out test set, 2,183 beats)
| | Threshold 0.5 | Validation-selected (0.217) |
|---|---|---|
| Balanced accuracy | 0.962 | 0.963 |
| Normal recall | 0.985 | 0.954 |
| Abnormal recall | 0.939 | 0.973 |
| ROC-AUC | 0.9955 | 0.9955 |

On computer-generated ECG images (several grid styles) the pipeline reached about 0.95 accuracy; on a harsher test with blur, noise, tilt and JPEG compression, Normal recall dropped to about 0.60. Real scans were **not** validated.

## Limitations
- The CSVs have **no patient IDs**; beats from one patient can appear in both train and test, so scores are likely optimistic. A patient-wise split is the next step.
- Image results come from generated images, not real clinical scans.
- Only Normal/Abnormal (PTBDB) is trained in the provided model; it cannot name a specific disease. Heart rate is computed separately from R-R spacing and assumes 25 mm/s paper.
- Class imbalance: PTBDB is about 72% Abnormal; MIT-BIH class N is about 83% of the data.

## Future work
Patient-wise splits, training on real labelled ECG images (2D CNN), multi-lead input, calibrated probabilities, 5-class MIT-BIH model for images.

## Acknowledgements and references
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE EMB Magazine, 2001.
- Bousseljot R, Kreiseler D, Schnabel A. PTB Diagnostic ECG Database (PhysioNet), 1995.
- Kachuee M, Fazeli S, Sarrafzadeh M. ECG Heartbeat Classification: A Deep Transferable Representation. IEEE ICHI, 2018.
- Kaggle "ECG Heartbeat Categorization Dataset" and PhysioNet.

