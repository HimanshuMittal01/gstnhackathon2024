# GSTN Analytics Hackathon 2024

> ### IMPORTANT NOTE:
> Use `submission/project-himanshu.zip` to verify the checksum: 734bfc240d3eed36de132aaccd4424f86f88adeb23afb102c0f0442e19210f75

Given the training and testing datasets, the objective is to build a model which predict the 'target' binary variable in the test dataset without any domain knowledge.

More details can be found in the
- [Concept doc](docs/concept.md)
- [Project report](docs/Project%20Report%20-%20GSTN%20Analytics%20Hackathon%202024.pdf)

The code achieves a performance with a **Average Precision (AP) score of 0.9407** and the following metrics:

- **PR-AUC**: 0.940762
- ROC-AUC: 0.995028
- Accuracy: 0.977403

<p align="center">
  <img src="docs/cm_test.png?raw=true" alt="Sublime's custom image" width=400 />
</p>

## How to run

Follow below steps to perform predictions on a new dataset:

You can clone the repo and zip the folder to verify the checksum or use 'Download ZIP' option in the GitHub.

*If you're cloning using CLI, please don't include any other data or model files while converting into zip to verify the checksum*


### 1. Setup virtual environment

Ensure you are using Python >= 3.12.

```bash
# Create a virtual environment
python -m venv venv

# Active the environment
# venv\Scripts\activate (On Windows)
source venv/bin/activate

# Install libraries
pip install -r requirements.txt
```

### 2. Download models

To perform inference, you need to download the trained models and artifacts. Alternatively, you can choose to train the model locally to generate these assets.

The following resources are saved on Google Drive:
- Pre-trained model 'stackEnsembleV2.joblib' (for classification) [Download here](https://drive.google.com/file/d/1Zz2T2_HJUC14Ebf0GWUKwYPpp4q1kWNw/view?usp=drive_link) (~400 MB)
- Isolation Forests 'isoforestsFS2.joblib' (for feature enginnering) [Download here](https://drive.google.com/file/d/1J6oS9HeL_IoS4OpgvgD9oU9h1CM91YRF/view?usp=drive_link) (~2 MB)
- Prediction (in csv) on the given test dataset [Download here](https://drive.google.com/file/d/168a5F4KMIDVOQCpZbvAfING0H-XlhoUH/view?usp=drive_link) (~10 MB)

If you choose to train locally, ensure you have the appropriate datasets and configurations as mentioned below in the project documentation.

*Note: The above links shall be accessible to anyone for the duration of the hackathon evaluation, presumably till Nov 1st, 2024*

### 3. Predict on test dataset

Recommended way to organize your data and assets:
```bash
gstnhackathon2024/
│
├── input/                         # Directory for datasets
│   ├── Train_60/
│   └── Test_20/
│
├── models/                        # Pre-trained models and artifacts
│   └── stackEnsembleV2.joblib
│   └── isoforestsFS2.joblib
│
├── output/                        # Store experiment results here
├── notebooks/                     # Jupyter Notebooks for experiments
│
├── train.py                       # Model architecture and training functions
├── infer.py                       # Inference code for using the model
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation
```

#### Sample usage for `infer.py`

```bash
# This will save predictions in 'output/predictions.csv' and evaluation metrics are returned in stdout
python infer..py \
--assets-filepath "models/isoforestsFS2.joblib" \
--model-filepath "models/stackEnsembleV2.joblib" \
--savepath "output/predictions.csv" \
--X "input/Test_20/X_Test_Data_Input.csv" \
--Y "input/Test_20/Y_Test_Data_Target.csv"
```

## Training and Performance

#### Sample usage for `train.py`

```bash
# This will save model and asset file in specified location
# It can take up to 2-10 minutes depending on the processor
python train..py \
--assets-filepath "models/isoforestsFS3.joblib" \
--model-filepath "models/stackEnsembleV3.joblib" \
--trainX "input/Train_60/X_Train_Data_Input.csv" \
--trainY "input/Train_60/Y_Train_Data_Target.csv" \
--testX "input/Test_20/X_Test_Data_Input.csv" \
--testY "input/Test_20/Y_Test_Data_Target.csv"
```

The provided model trained using this script achieves the following evaluation metrics:

**Average Metrics**

| Metric  | Macro Avg      | Weighted Avg  |
|-------------|----------------|---------------|
| Precision | **0.935594**  | 0.977301      |
| Recall    | 0.931509      | 0.977403      |
| F1-Score  | 0.933539      | 0.977349      |

**Classification Report**

| Class | Precision | Recall   | F1-Score |
|-------|-----------|----------|----------|
|   0   | 0.986995  | 0.988069 | 0.987532 |
|   1   | 0.884193  | 0.874949 | 0.879547 |
