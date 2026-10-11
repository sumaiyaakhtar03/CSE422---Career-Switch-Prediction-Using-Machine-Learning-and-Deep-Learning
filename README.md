# Career Switch Prediction Using Machine Learning and Deep Learning

A binary classification project that explores employee and career-related information to predict whether a person is likely to change careers. The project compares traditional machine learning classifiers with a feed-forward neural network, while investigating class imbalance and preprocessing strategies.

## Project overview

The notebook `Copy_of_422_PROJECT.ipynb` works with **5,000 records** from `Career_Switch_Prediction_Dataset.csv`. Its prediction target is `will_change_career`:

- **0:** Will not change career
- **1:** Will change career

The notebook reports a class distribution of **74.76%** for class 0 and **25.24%** for class 1, making class imbalance an important part of the analysis.

## Key features

- Exploratory data analysis (EDA) for numerical and categorical variables
- Descriptive statistics, distributions, histograms, boxplots, density plots, and correlation heatmaps
- Pearson, Spearman, and Kendall correlation analyses against the target
- Missing-value treatment for numeric and categorical features
- Label encoding, one-hot encoding, experience mapping, and city target-mean encoding
- Feature standardization with `StandardScaler`
- Stratified **70% training / 30% testing** split
- **SMOTEENN** hybrid resampling on training data to address class imbalance
- Comparison of **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Logistic Regression**
- Feed-forward **Artificial Neural Network (ANN)** using TensorFlow/Keras
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC curves, and training loss

## Dataset

**Required file:** `Career_Switch_Prediction_Dataset.csv`

The notebook uses career-related variables including:

- City and city development index
- Gender and relevant experience
- University enrollment, education level, and major discipline
- Years of experience and training hours
- Company size, company type, and time since last job change
- `will_change_career` (prediction target)

> The CSV is loaded from `/content/Career_Switch_Prediction_Dataset.csv` in the supplied notebook. The dataset must be obtained and placed at that location (or the file path updated). This README does not assume the CSV is included in the repository.

## Models

| Model | Approach |
| --- | --- |
| K-Nearest Neighbors | Distance-based classification |
| Decision Tree | Rule-based tree classification |
| Logistic Regression | Linear probabilistic classification |
| Artificial Neural Network | Dense layers: 128 → 64 → 1, with ReLU and sigmoid activations |

The ANN uses the Adam optimizer (learning rate 0.001), binary cross-entropy loss, **30 epochs**, and a **batch size of 128**. The three classical classifiers are trained on SMOTEENN-resampled training data; the ANN is trained on the original split's training data.

## Results

The saved notebook output reports the following for the ANN:

| Metric | Value |
| --- | --- |
| ANN test accuracy | **76.73%** |

The notebook also produces comparative metric plots, confusion matrices, and ROC curves for KNN, Decision Tree, and Logistic Regression. Exact numeric comparison values are not printed in the saved notebook output, so they are not reproduced here. Run the notebook to inspect the full comparisons.

## Technologies

- Python
- Jupyter Notebook / Google Colab
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- imbalanced-learn
- TensorFlow / Keras

## Getting started

### Option 1: Google Colab

1. Open `Copy_of_422_PROJECT.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `Career_Switch_Prediction_Dataset.csv` to Colab's `/content/` directory using the Files panel.
3. Select **Runtime → Run all**.
4. Review the plots, model comparisons, and neural-network training output.

### Option 2: VS Code / Jupyter locally

1. Install Python and the Python and Jupyter extensions for VS Code.
2. Clone the repository and open the project folder.
3. Create and activate a virtual environment:

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell)**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install dependencies:
   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow notebook ipykernel
   ```
5. Open `Copy_of_422_PROJECT.ipynb` and select your virtual environment as the notebook kernel.
6. Place `Career_Switch_Prediction_Dataset.csv` in the project folder and replace the Colab-specific loading path:
   ```python
   career = pd.read_csv('Career_Switch_Prediction_Dataset.csv')
   ```
7. Run notebook cells from top to bottom.

## Project structure

```text
career-switch-prediction/
├── Copy_of_422_PROJECT.ipynb
├── Career_Switch_Prediction_Dataset.csv  # required dataset, if redistribution is permitted
└── README.md
```

## Important considerations

- The current notebook computes some preprocessing transformations, including standardization and city target-mean encoding, **before** splitting into training and testing sets. This can introduce data leakage and inflate measured performance. For rigorous evaluation, fit each transformation using only the training data and apply it to the test data afterward. Target encoding should be fitted out-of-fold on training data.
- SMOTEENN is applied to training data only, which helps prevent resampling-related test leakage.
- The neural network is not trained with the same SMOTEENN-resampled data as the classical models, so model comparisons should account for that difference.
- Prediction is based on patterns in this dataset and should not be treated as a definitive career decision or employment recommendation.

## Future improvements

- Build a leakage-safe preprocessing and modeling pipeline
- Compare all models using a consistent evaluation protocol and cross-validation
- Tune hyperparameters and investigate threshold selection for the minority class
- Report complete numerical test metrics and calibration
- Add a simple web interface for demonstration, if appropriate

## License

No license is specified here. Add a `LICENSE` file if you want to define how others may use or redistribute the project.
