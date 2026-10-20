# Person A — FER2013 assignment

This is simple starter code for our FER2013 baseline and creative challenge. It needs to be run on the FER2013 data to produce results; there are **no made-up scores**.

## Put files in the team repository
Copy `person_a_experiments.py`, `requirements_person_a.txt`, and the `src/features.py` file to the repository root, keeping the `src/` subfolder. **Review before replacing your team's shared `src/features.py`** because other teammates might be using it. Do not commit large datasets or model files without team agreement.

## Data needed
Download FER2013 and use the team's `z_load_split_fer2013.ipynb` to create:

- `data/fer/fer2013_train.csv`
- `data/fer/fer2013_validation.csv`
- `data/fer/fer2013_test.csv`

Each file must contain `emotion` (0 to 6) and `pixels` columns. If your notebook saves the splits elsewhere, pass `--data-dir PATH`.

## Run from the repository root

```powershell
python -m pip install -r requirements_person_a.txt
python person_a_experiments.py --max-train 6000 --max-validation 1500
```

For a quicker smoke test, try `--max-train 1000 --max-validation 300`. Change these sizes for your real experiment. SVM and dlib landmarks can be slow. The script uses only a stratified sample of the training/validation images by default; say this in your report if you keep this sampling.

When you have selected your approach using validation results, run the **one-time** final evaluation:

```powershell
python person_a_experiments.py --max-train 6000 --max-validation 1500 --final-test
```

That run refits the candidate models on the same training subset and selects by validation again. For strict test-set separation, make no changes based on the final test metrics.

## What it makes
- `results/person_a/baseline_validation.csv` — comparison across HOG / landmarks / PCA and LR / SVM / RF
- `results/person_a/creative_ablation_validation.csv` — baseline, class balancing, parameter change, both
- `results/person_a/*.png` — class counts, example faces, confusion matrices
- `models/person_a/*.pkl` — fitted models, feature metadata

**Limitations:** This is a starting point, not a completed study. The SVM is RBF only, the ablation is a small check, and there is no systematic hyperparameter search. PCA uses 100 components for speed and is fitted only on training data (record the explained variance in a more complete study). The script does not yet collect hard-case images, and saved models should not be treated as standalone webcam pipelines unless feature extraction is integrated in the demo. The final test should ideally be run only after all validation decisions are frozen.
