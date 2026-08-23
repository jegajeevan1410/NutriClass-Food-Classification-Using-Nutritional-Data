# NutriClass: Food Classification Using Nutritional Data

A machine learning project that predicts the **exact food name** from a nutritional profile
(calories, protein, fat, carbs, sugar and more) using multi-class classification.

**Domain:** Food and Nutrition / Machine Learning
**Skills:** Data Preprocessing, Feature Engineering, Multi-class Classification, Evaluation Metrics

---

## Problem Statement

In an era of increasing dietary awareness, the ability to classify food items by their
nutritional attributes is valuable. This project builds a machine learning model that takes
tabular nutritional data as input and predicts which food it belongs to, along with insights
into what makes each food category distinct.

---

## Business Assumption

The **target variable is the food name**, not a broad food category.

In a strict diet plan, a user specifies the nutritional requirements they want for a meal
(for example, high-protein and low-sugar for dinner). Those metrics are fed to the model,
which returns the **one exact food** from the allowed set that meets the target.

This is why the problem is framed as **classification rather than recommendation**. A
recommender would suggest five alternatives; a strict diet plan needs a single precise
choice, giving a one-to-one mapping between a nutrition profile and an allowed food.

**Business value:** improves diet adherence, speeds up meal selection, removes manual
nutrition checking, and supports personalised diet tracking in real-time meal-logging apps.

---

## Dataset

`synthetic_food_dataset_imbalanced.csv`

| Property | Value |
|---|---|
| Rows | 31,700 |
| Feature columns | 15 |
| Target column | `Food_Name` |
| Classes | 10 |
| Missing values | 375 per numeric column |
| Duplicate rows | 313 |

### Columns

**Numerical (11):** Calories, Protein, Fat, Carbs, Sugar, Fiber, Sodium, Cholesterol,
Glycemic_Index, Water_Content, Serving_Size

**Categorical (2):** Meal_Type (breakfast/lunch/dinner/snack), Preparation_Method
(raw/fried/baked/grilled)

**Boolean (2):** Is_Vegan, Is_Gluten_Free

### Class distribution — the dataset is imbalanced

| Food | Samples | Food | Samples |
|---|---|---|---|
| Pizza | 6,000 | Ice Cream | 3,000 |
| Burger | 5,000 | Steak | 2,000 |
| Donut | 4,500 | Apple | 1,500 |
| Pasta | 4,000 | Banana | 1,200 |
| Sushi | 3,500 | Salad | 1,000 |

The largest class has **6× more samples** than the smallest. This is handled with a
stratified train/test split and weighted evaluation metrics, and a per-class report is
printed for every model so weak performance on rare classes cannot hide behind a high
overall accuracy.

---

## Tech Stack

- **Python 3**
- **pandas**, **numpy** — data handling
- **matplotlib**, **seaborn** — visualisation
- **scikit-learn** — preprocessing, models, metrics
- **xgboost** — gradient boosting

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd nutriclass
   ```

2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```

3. Place `synthetic_food_dataset_imbalanced.csv` in the same folder as the notebook.

4. Launch Jupyter and run all cells top to bottom:
   ```bash
   jupyter notebook NutriClass_Food_Classification.ipynb
   ```

No code edits are needed — the file path and target column are already set.

**Runtime note:** most cells run in seconds. Gradient Boosting takes about **100 seconds**
(scikit-learn builds 100 trees per class across 10 classes) and cross-validation takes
about **25 seconds**. This is expected.

---

## Project Structure

```
nutriclass/
├── NutriClass_Food_Classification.ipynb   # main notebook
├── synthetic_food_dataset_imbalanced.csv  # dataset
└── README.md
```

---

## Workflow

| Step | What happens |
|---|---|
| 1 | Import libraries, set a fixed random seed for reproducibility |
| 2 | Load the dataset |
| 3 | Explore: data types, statistics, class distribution, outliers, correlations |
| 4 | Preprocess: fill missing values, drop duplicates, cap outliers, encode, scale |
| 5 | Feature engineering: label-encode the target, apply PCA |
| 6 | Stratified 80/20 train/test split |
| 7 | Train and evaluate 7 classifiers, one at a time |
| 8 | Compare all models |
| 9 | 5-fold cross-validation check |
| 10 | Feature importance |
| 11 | Business assumption in action — predict a food from a nutrition goal |
| 12 | Insights and conclusion |

### Preprocessing decisions

- **Missing values** → filled with the column **median**, which outliers do not distort.
- **Duplicates** → removed, so the same row cannot appear in both train and test.
- **Outliers** → **capped** using the IQR rule rather than deleted, preserving the rows.
- **Booleans** → converted to 1/0.
- **Categorical** → **one-hot encoded**, because Meal_Type and Preparation_Method have no
  natural order; numbering them 0–3 would falsely tell the model that dinner > lunch.
- **Scaling** → StandardScaler, essential for KNN, SVM and Logistic Regression, which
  compare distances.
- **PCA** → 21 encoded features reduced to 12 components, retaining 95% of the variance.

---

## Evaluation Metrics

Every model is measured on all five:

| Metric | Meaning |
|---|---|
| **Accuracy** | Percentage of correct predictions overall |
| **Precision** | Of everything predicted as a food, how much really was that food |
| **Recall** | Of all actual samples of a food, how many the model found |
| **F1-score** | The balance between precision and recall |
| **Confusion Matrix** | Where the model was right and where it confused foods |

Precision, recall and F1 use `average='weighted'`, which scores each food separately and
averages by class size — the appropriate choice for imbalanced data.

**Reading the confusion matrix:** rows are the actual food, columns are the predicted food,
and the diagonal holds correct predictions. For any single class, TP is its diagonal cell,
FP is the rest of its column, FN is the rest of its row, and TN is everything else.

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Support Vector Machine** | **98.79%** | 0.9879 | 0.9879 | 0.9879 |
| XGBoost | 98.68% | 0.9868 | 0.9868 | 0.9868 |
| Random Forest | 98.49% | 0.9849 | 0.9849 | 0.9849 |
| Gradient Boosting | 98.45% | 0.9846 | 0.9845 | 0.9845 |
| Logistic Regression | 98.44% | 0.9844 | 0.9844 | 0.9844 |
| K-Nearest Neighbors | 98.44% | 0.9844 | 0.9844 | 0.9844 |
| Decision Tree | 97.55% | 0.9755 | 0.9755 | 0.9755 |

**Best model:** Support Vector Machine — 98.79% accuracy

**5-fold cross-validation (Random Forest):** mean 98.35%, standard deviation 0.0035.
The low spread confirms the result is stable rather than a lucky split.

---

## Business Assumption in Action

Step 11 implements the **User-Defined Nutritional Goals** assumption as a working function:

```python
user_goals = {
    'Calories': 400, 'Protein': 25, 'Fat': 25, 'Carbs': 2,
    'Sugar': 1, 'Fiber': 0.5, 'Sodium': 72, 'Cholesterol': 90,
    'Glycemic_Index': 2, 'Water_Content': 56, 'Serving_Size': 200
}

predict_food(user_goals, meal_type='dinner', preparation_method='grilled')
# → 'Steak'
```

The user's input travels the **same pipeline** the training data did —
`input → scaler → PCA → model → food name`. Skipping either transform would hand the model
numbers in a shape it never learned on.

**Sample results**

| Goal | Meal | Prediction |
|---|---|---|
| High protein, very low carb, low GI | dinner | Steak |
| Low calorie, low sodium, high water content | snack | Banana |
| Light, low GI, high water content | lunch | Salad |

---

## Insights

- **Ensemble and margin-based models perform best.** SVM and XGBoost lead, while the single
  Decision Tree trails by roughly a point — the relationship between a nutrition profile and
  a food name is not linear, and a lone tree overfits where an ensemble does not.
- **Every model clears 97%.** All ten foods are well separated in nutritional space, so even
  Logistic Regression handles the task.
- **Rare classes are predicted well.** Despite the 6:1 imbalance, Salad (1,000 samples)
  scores around 0.99 F1 — the stratified split kept every class properly represented.
- **The confusion matrix concentrates its errors on Pizza and Burger**, which share similar
  calorie, fat and sodium profiles.
- **Glycemic_Index and Water_Content are highly discriminative.** Steak sits near GI 1 while
  Pizza sits near 80; fruits carry 75–86% water against 30–45% for baked items.

---

## Limitations

- **The dataset is synthetic.** 98%+ across every model, including plain Logistic Regression,
  indicates cleanly separated classes. Real nutrition data would overlap far more and these
  scores should not be read as a real-world expectation.
- **The model always returns an answer.** Given an impossible profile — 200g of protein at
  zero calories — it still returns the nearest food it knows. A production system would add
  a confidence threshold and reply "no matching food found."
- **Unspecified features fall back to medians**, which biases the prediction. The median
  sodium is ~292mg, a savoury value, so a fruit query that omits sodium is pushed away from
  fruits. The more of the profile supplied, the better the prediction.
- **Apple and Banana overlap closely** (101 vs 110 calories, GI 41 vs 53) and are the two
  classes most often swapped.
- **No hyperparameter tuning** was performed; all models use their default or near-default
  settings.

---

## Project Deliverables

- [x] **Source code** — documented Jupyter notebook
- [x] **Report** — approach, data analysis, model selection and evaluation, insights
- [x] **Visualisations** — class distribution, box plots, correlation heatmap, per-class
      comparisons, PCA variance, confusion matrices, model comparison, feature importance

---

## References

Dataset and project brief provided by **GUVI / HCL** as part of the Data Science & AI/ML
capstone programme.
