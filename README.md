# ML Practice

A hands-on repository where I implement, test, and document various machine learning algorithms and workflows. Covers exploratory data analysis (EDA), preprocessing, classification, regression, clustering, dimensionality reduction, ensemble methods, and NLP.

---

## Notebooks at a Glance

| Notebook | Task | Dataset | Models / Algorithms |
|---|---|---|---|
| [`student-placement.ipynb`](student-placement.ipynb) | Binary Classification | Student Placement (100k rows) | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| [`TitanicModels.ipynb`](TitanicModels.ipynb) | Binary Classification | Titanic (Seaborn) | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM |
| [`HeartdiseaseFinal.ipynb`](HeartdiseaseFinal.ipynb) | Binary Classification | Heart Disease (`heart.csv`) | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM |
| [`Heart.ipynb`](Heart.ipynb) | EDA & Preprocessing | Heart Disease (`heart.csv`) | Data cleaning and visualization scratchpad |
| [`ford-car-price.ipynb`](ford-car-price.ipynb) | Regression | Ford Used Cars | Linear Regression (One-Hot vs Label Encoding) |
| [`insurance.ipynb`](insurance.ipynb) | Regression | Medical Insurance (`insurance.csv`) | Linear Regression + Pearson & Chi-Square tests |
| [`clustering.ipynb`](clustering.ipynb) | Unsupervised | Synthetic Blobs & Moons | K-Means (Elbow method) vs DBSCAN |
| [`PCADimensions.ipynb`](PCADimensions.ipynb) | Dimensionality Reduction | Synthetic Blobs (5D) | PCA (Principal Component Analysis) |
| [`EnsembleLearning.ipynb`](EnsembleLearning.ipynb) | Multiclass Classification | Iris (Seaborn) | Stacking, Random Forest, AdaBoost, Gradient Boosting, XGBoost |
| [`GridSearchCV.ipynb`](GridSearchCV.ipynb) | Hyperparameter Tuning | Iris (Seaborn) | SVM tuning (GridSearchCV vs RandomizedSearchCV) |
| [`emotion-based-on-text.ipynb`](emotion-based-on-text.ipynb) | NLP / Text Classification | Emotion Dataset (`train.txt`) | Naive Bayes & Logistic Regression (BoW vs TF-IDF) |

---

## Detailed Notes on Each Project

### 1. Student Placement Prediction (`student-placement.ipynb`)
My most complete notebook here. Analyzes 100,000 student records across features like CGPA, IQ, internships, projects, communication skills, and backlogs to predict campus placement.
- Features engineered: a weighted composite score combining academics, IQ, and skills.
- Models tested: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Evaluated with 5-fold Stratified Cross-Validation, ROC-AUC curve, confusion matrix, and feature importances.
- Includes a helper function at the end (`predict_placement`) to test predictions on new student inputs, and saves the trained model with `joblib`.

### 2. Titanic Survival (`TitanicModels.ipynb`)
A benchmark of classic classification models on the passenger dataset.
- Drops high-null/redundant columns (`deck`, `alive`, `who`, etc.), imputes missing age with the mean, and encodes categorical features.
- Trains and compares Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, and linear SVM.
- Tests 5-fold cross-validation on SVM.

### 3. Heart Disease Classification (`HeartdiseaseFinal.ipynb` & `Heart.ipynb`)
Predicting presence of heart disease based on clinical indicators.
- Handles impossible zero values in resting blood pressure and serum cholesterol by imputing with non-zero means.
- One-hot encodes categorical variables and standardizes numerical features with `StandardScaler`.
- Compares Logistic Regression, KNN, Naive Bayes, Decision Tree, and SVM (RBF kernel).
- Exports the trained model and scaler to `.pkl` files.
- *Note:* `Heart.ipynb` was my initial exploratory/data-cleaning draft; `HeartdiseaseFinal.ipynb` has the full modeling pipeline.

### 4. Ford Car Price Prediction (`ford-car-price.ipynb`)
Predicting used Ford car prices using mileage, year, engine size, fuel type, and transmission.
- Runs EDA with distribution plots and correlation heatmaps.
- Compares model performance when using One-Hot Encoding (`pd.get_dummies`) versus Label Encoding on the categorical features.
- Evaluates $R^2$ and Adjusted $R^2$.

### 5. Medical Insurance Cost Prediction (`insurance.ipynb`)
Predicting individual medical charges billed by health insurance.
- Performs EDA with boxplots and histograms across age, BMI, and charges.
- Applies statistical checks: Pearson correlation for continuous variables and Chi-Square contingency tests on binned charges for categorical variables.
- One-hot encodes region, bins BMI into categories, scales continuous features, and fits a Linear Regression model.

### 6. K-Means vs DBSCAN Clustering (`clustering.ipynb`)
A quick visual comparison of centroid-based vs density-based clustering.
- Part 1: Uses synthetic blobs with K-Means, plotting the Elbow curve (inertia) to pick the optimal $K=3$.
- Part 2: Compares K-Means vs DBSCAN on non-spherical data (`make_moons`). Shows where K-Means fails on interleaved crescents and how DBSCAN separates arbitrary shapes naturally without needing $K$ specified upfront.

### 7. PCA Dimensionality Reduction (`PCADimensions.ipynb`)
Demonstrates how Principal Component Analysis reduces higher-dimensional data down to 2 dimensions for visualization.
- Generates 5-dimensional synthetic blob data.
- Standardizes features and applies `PCA(n_components=2)`.
- Plots the 2D projection (PC1 vs PC2) with Seaborn.

### 8. Ensemble Learning (`EnsembleLearning.ipynb`)
Testing different ensemble paradigms on the Iris dataset.
- Stacking: Combines Decision Tree, RBF SVM, and Logistic Regression as base learners, fed into a Logistic Regression meta-learner.
- Bagging / Forest: `RandomForestClassifier`.
- Boosting: `AdaBoostClassifier`, `GradientBoostingClassifier`, and `XGBClassifier`.

### 9. Hyperparameter Tuning (`GridSearchCV.ipynb`)
Demonstrates systematic tuning for an SVM classifier.
- Compares `GridSearchCV` (exhaustive search over candidate $C$ and kernel values) against `RandomizedSearchCV` (sampling a set number of iterations).
- Shows how to convert `cv_results_` into a Pandas DataFrame for easy sorting and inspection.

### 10. Emotion Classification from Text (`emotion-based-on-text.ipynb`)
Basic NLP pipeline classifying raw text sentences into emotion categories.
- Preprocessing: lowercasing, punctuation removal, digit stripping, non-ASCII cleanup, and NLTK stopword removal.
- Compares Bag-of-Words (`CountVectorizer`) vs TF-IDF (`TfidfVectorizer`).
- Trains Multinomial Naive Bayes and Logistic Regression.

---

## Datasets

Some notebooks use built-in datasets from `seaborn` (`iris`, `titanic`) or `sklearn.datasets` (`make_blobs`, `make_moons`), so they run without extra files.

For the notebooks that require CSV/text files:
- **`placement.csv`**: Used in `student-placement.ipynb` (from [Kaggle Student Placement Dataset](https://www.kaggle.com/datasets/sahil2604singh/student-placement-dataset)).
- **`heart.csv`**: Used in `HeartdiseaseFinal.ipynb` and `Heart.ipynb` (from [Kaggle Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)).
- **`insurance.csv`**: Used in `insurance.ipynb` (from [Kaggle Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)).
- **`ford.csv`**: Used in `ford-car-price.ipynb` (from [Kaggle Ford Car Price Prediction](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction)).
- **`train.txt`**: Used in `emotion-based-on-text.ipynb` (semicolon-separated text/emotion dataset).

*Tip:* When running locally, place these files in the root folder or a `data/` directory.

---

## How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Sahil26singh/ML-practice.git
   cd ML-practice
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

---

## Practical Lessons & Gotchas

A few key takeaways and common pitfalls to keep in mind across these workflows:

1. **Watch out for test-set leakage when scaling:**
   Never call `scaler.fit_transform()` on test data or on the full dataset before splitting. Always fit the scaler strictly on the training set, and only call `scaler.transform()` on the test set. Wrapping steps in an `sklearn.pipeline.Pipeline` helps prevent this mistake automatically.

2. **Don't use `df = df.astype(int)` on entire DataFrames:**
   It is tempting to run `df.astype(int)` to convert True/False dummy variables into 0 and 1, but doing so converts all columns. This silently cuts off decimals in continuous features (like `Oldpeak` in heart data, `bmi` and `charges` in insurance, or `engineSize` in car data). A cleaner way is using `pd.get_dummies(..., dtype=int)`.

3. **Accuracy is not enough for medical / imbalanced data:**
   In datasets like heart disease, a False Negative (missing someone who has disease) is far worse than a False Positive. Always look at **Recall** (Sensitivity), Confusion Matrix, and ROC-AUC alongside basic accuracy.

4. **Feature interactions matter in linear models:**
   In the insurance dataset, linear regression alone hits a ceiling ($R^2 \approx 0.75$). But adding an interaction term between `smoker` and `bmi` gives the model the ability to capture exponential risk for obese smokers, bumping performance significantly.
