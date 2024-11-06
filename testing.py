import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif     
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold,
    cross_validate,
    learning_curve
)
from sklearn.metrics import (
    classification_report,
    RocCurveDisplay,
    confusion_matrix
)


longitudinal_df = pd.read_csv('patient_longitudinal.csv', delimiter='\t')

# print(longitudinal_df)
# print(longitudinal_df.describe())
# print(longitudinal_df.info())

# print(longitudinal_df.patient_id.unique())

longitudinal_df.sex = longitudinal_df.sex.astype(int)
longitudinal_df.patient_id = longitudinal_df.patient_id.astype(int)
longitudinal_df.smoking = longitudinal_df.smoking.astype(int)
longitudinal_df.diabetes = longitudinal_df.diabetes.astype(int)
longitudinal_df.visit_date = pd.to_datetime(longitudinal_df.visit_date, format='%Y-%m-%d', errors='coerce')

longitudinal_df['visit_date'] = pd.to_datetime(longitudinal_df['visit_date'], format='%Y-%m-%d')

bp_series = longitudinal_df.set_index('visit_date')['bp_systolic']

bp_series = bp_series.groupby(level=0).mean() #take mean of duplicated days
bp_series = bp_series.interpolate()  # fill missing values

# # 2. Analyze blood pressure trends

monthly_bp = bp_series.resample('M').mean()
monthly_bp = monthly_bp.interpolate()
monthly_bp_smooth = monthly_bp.rolling(window=3, min_periods=1).mean()
# print(monthly_bp_smooth)

# plt.figure(figsize=(12, 6))
# plt.plot(monthly_bp, label='Rolling Average')
# plt.plot(monthly_bp_smooth, label='3-Month Rolling Average')
# plt.title('BP Trends')
# plt.legend()
# plt.show()

baseline_df = pd.read_csv('patient_baseline.csv', delimiter='\t')

#fix datatypes
baseline_df.sex = baseline_df.sex.astype(int)
baseline_df.smoking = baseline_df.smoking.astype(int)
baseline_df.diabetes = baseline_df.diabetes.astype(int)
# print(baseline_df.describe())
# print(baseline_df.info())

# 1. Analyze factors affecting baseline blood pressure:
#    - Use statsmodels OLS to predict `bp_systolic`
#    - Include `age`, `bmi`, `smoking`, and `diabetes` as predictors
#    - Interpret the coefficients and their p-values
#    - Assess model fit using R-squared and diagnostic plots
#    - Tips:
#      - Create feature matrix `X` with predictors and add constant term using `sm.add_constant()`
#      - Use `sm.OLS(y, X).fit()` to fit the model
#      - Use `summary()` to examine p-values and confidence intervals
#      - Plot residuals vs fitted values and Q-Q plot
#      - Consider robust standard errors with `HC3` covariance type



# 1. Analyze factors affecting baseline blood pressure
X = sm.add_constant(baseline_df[['age', 'bmi', 'smoking', 'diabetes']])
y = baseline_df['bp_systolic']

model = sm.OLS(y, X).fit(cov_type='HC3')
# print(model.summary())

#R^2 = 0.376
                #  coef    std err          t      P>|t|      [0.025      0.975]
# age            0.2840      0.023     12.551      0.000       0.240       0.328
# bmi            0.2786      0.066      4.198      0.000       0.148       0.409
# smoking        5.2412      0.698      7.512      0.000       3.872       6.610
# diabetes       9.8732      0.742     13.307      0.000       8.417      11.329


# # Diagnostic plots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # residuals vs. fitted
# ax1.scatter(model.fittedvalues, model.resid)
# ax1.set_xlabel('Fitted values')
# ax1.set_ylabel('Residuals')
# ax1.set_title('Residuals vs Fitted Plot')
# ax1.axhline(y=0, color='r')

# #qq plot
# sm.graphics.qqplot(model.resid, fit=True, line='45', ax=ax2)
# ax2.set_title('Q-Q Plot of Residuals')
# plt.tight_layout()
# plt.show()

# 2. Model treatment effectiveness:
#    - Fit a GLM with binomial family to predict treatment success
#    - Use baseline characteristics and `adherence` as predictors
#    - Report odds ratios and their confidence intervals
#    - Assess model fit using deviance and diagnostic plots
#    - Tips:
#      - Create feature matrix `X` with predictors and add constant term
#      - Use `sm.GLM(y, X, family=sm.families.Binomial()).fit()`
#      - Get odds ratios with `np.exp(params)`
#      - Check residual deviance vs null deviance
#      - Use `influence()` to detect influential observations

treatment_df = pd.read_csv('patient_treatment.csv', delimiter='\t') #looks fine no fixing needed
# print(treatment_df.info())
# print(treatment_df.describe())


X = sm.add_constant(treatment_df[['age', 'sex', 'bmi', 'smoking', 'bp_systolic', 'cholesterol', 
                                    'heart_rate', 'diabetes', 'adherence']])
y = treatment_df['outcome']
glm_model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
# print(glm_model.summary())
# odds_ratios = np.exp(glm_model.params)
# print("Odds Ratios:")
# print(odds_ratios)
# print("CIs")
# print(np.exp(glm_model.conf_int()))

#Deviance plot
# plt.figure(figsize=(10, 6))
# plt.scatter(glm_model.fittedvalues, glm_model.resid_deviance)
# plt.xlabel('Fitted values')
# plt.ylabel('Deviance residuals')
# plt.title('Deviance Residuals vs Fitted Values')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.show()

# Q-Q Plot of Deviance 
# fig, ax = plt.subplots(figsize=(10, 6))
# sm.graphics.qqplot(glm_model.resid_deviance, line='45', ax=ax)
# plt.title('Q-Q Plot of Deviance Residuals')
# plt.show()


# print("Null Deviance:", glm_model.null_deviance) #1331.98
# print("Residual Deviance:", glm_model.deviance) #1305.81

#influential points
cooks_d = glm_model.get_influence().cooks_distance[0]
cooks_d_series = pd.Series(cooks_d, index=X.index)
# print("\nTop 5 influential observations:")
# print(cooks_d_series.nlargest(5))


# 1. Build a prediction pipeline:
#    - Create features from baseline characteristics
#    - Standardize numeric features using `StandardScaler`
#    - Train a logistic regression model to predict treatment outcomes
#    - Include regularization to prevent overfitting
#  - Use `ColumnTransformer` for mixed numeric/categorical features
#  - Consider `SelectKBest` or `RFE` for feature selection
#  - Try different regularization strengths with `C` parameter
#  - Use `Pipeline` to prevent data leakage


X = treatment_df.drop(['patient_id', 'adherence', 'outcome', 'treatment'], axis=1)
y = treatment_df['outcome']

# Identify numeric and categorical columns
categorical_features = X.loc[:, ['sex', 'smoking', 'diabetes']].columns
# print(categorical_features)
numeric_features = X.drop(columns=categorical_features).columns
# print(numeric_features)

# preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# pipeline
pipeline = make_pipeline(
    preprocessor,
    SelectKBest(f_classif, k=5),
    LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
)

# 2. Validate model performance:
#    - Split data into 70% training and 30% test sets
#    - Implement 5-fold cross-validation on the training set
#    - Report accuracy, precision, recall, and ROC AUC
#    - Generate confusion matrix and ROC curve
    # - Use `StratifiedKFold` for imbalanced datasets
    # - Consider precision-recall curve for imbalanced data
    # - Plot learning curves to diagnose bias/variance
    # - Use `cross_validate` for multiple metrics at once


# Stratified split for imbalanced data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y,
    random_state=42
)

# Cross-validation with multiple metrics
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    pipeline,
    X_train, y_train,
    cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'roc_auc']
)

# Print cross-validation results
print("Cross-validation results:")
for metric, values in scores.items():
    if metric.startswith('test_'):
        print(f"{metric[5:]}: {values.mean():.3f}")

# Fit the model on the entire training set
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# Plot ROC curve
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
plt.title('ROC Curve')
plt.show()

#Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='roc_auc'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('ROC AUC Score')
plt.title('Learning Curves')
plt.legend()
plt.show()
