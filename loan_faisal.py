import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv('loan_dataset.csv')

#print(df.columns)
#print(df.head())

# EDA

f = pd.read_csv('loan_dataset.csv')

#print(df.columns)
#print(df.head())

# EDA
# Shows number of rows and columns
#df.shape()

# Shows columns with data types adn null vales
#df.info()
print(df.info())

# shows 
print(df.describe())

print(df.isnull().sum())

print(df.columns)

numeric_columns = ['age','monthly_income_pkr', 'credit_score', 'loan_amount_pkr',
       'loan_tenure_months', 'existing_loans']

for col in numeric_columns:
    print(f"Saving graph for {col}")
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    
    # This saves the file to your folder instead of showing a window
    plt.savefig(f"{col}_distribution.png") 
    plt.close() # Closes the plot to save memory


###### GENDER GRAPH
sns.countplot(x=df['gender'])

# 1. Clean the data (remove any weird spaces or NaNs)
df['gender'] = df['gender'].astype(str).str.strip()

# 2. Clear any previous "stuck" plots
plt.clf() 
plt.close('all')

# 3. Create the plot with a defined figure
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='gender', palette='viridis')

# 4. Add labels (helps the backend recognize the axis)
plt.title("Gender Distribution")
plt.xlabel("Gender Category")
plt.ylabel("Count")

# 5. Force the window to pop up
plt.show(block=True)

sns.countplot(x=df['gender'])
plt.savefig('gender_plot.png')
print("Graph saved as gender_plot.png - check your folder!")

#### GRAPH TO CHECK CITIES
sns.countplot(x=df['city'])
# 1. Clean the data (remove any weird spaces or NaNs)
df['city'] = df['city'].astype(str).str.strip()

# 2. Clear any previous "stuck" plots
plt.clf() 
plt.close('all')

# 3. Create the plot with a defined figure
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='city', palette='viridis')

# 4. Add labels (helps the backend recognize the axis)
plt.title("city Distribution")
plt.xlabel("city Category")
plt.ylabel("Count")

# 5. Force the window to pop up
plt.show(block=True)

sns.countplot(x=df['city'])
plt.savefig('city_plot.png')
print("Graph saved as city_plot.png - check your folder!")

################ BOX PLOTS TO CHECK OUTLIERS

numeric_columns = ['age','monthly_income_pkr', 'credit_score', 'loan_amount_pkr',
       'loan_tenure_months', 'existing_loans']

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.savefig(f"{col}_boxplot.png") 
    plt.close() # Closes the plot to save memory

    ############## HEAT MAP
    # Set figure size
plt.figure(figsize=(10, 8))

# Calculate correlation first, then pass to heatmap
corr_matrix = df.corr(numeric_only=True)

# Pass annot=True to the heatmap function, not the corr function
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")

plt.title("Correlation Heatmap of Loan Features")
plt.savefig("heatmap_loan.png")
plt.show()

######## DATA CLEANSING
df_cleaned = df.copy()
df_cleaned.head()
print("clean data copy")
print(df_cleaned.head())


### Remove Duplicates
df_cleaned.shape
df_cleaned.drop_duplicates(inplace=True)
print("Duplicate Removed")

df_cleaned.shape

### Remove Null
df_cleaned.isnull().sum()
print(df_cleaned.isnull().sum())

### Convert Text to Numbers Gender

df_cleaned['gender'].value_counts()
print(df_cleaned['gender'].value_counts())

# Convert gender to numbers: M = 1, F = 0

df_cleaned['gender'] = df_cleaned['gender'].map({'M': 1, 'F': 0})

# 1. Define your Target (what you want to predict)
y = df_cleaned['approved']

# 2. Define your Features (everything except the target)
X = df_cleaned.drop(columns=['approved','applicant_name'])

df_cleaned.drop(columns=['approved','applicant_name'])



#print(f"cleaned data drop approved and applicant name {df_cleaned.head()}")

# 'y' is your target (approved), 'gender' is the feature
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

### Convert Text to Numbers City
df['city'].value_counts()
print(df['city'].value_counts())

### One-Hot Encoding applied to City
df_cleaned = pd.get_dummies(df_cleaned, columns=['city'], prefix='city')
print(df_cleaned.head())


####### Employement TYpe checking
df['employment_type'].value_counts()
print(df['employment_type'].value_counts())


# Since there is a logical "hierarchy" to employment (e.g., being Salaried or a Business Owner usually carries different weight than being Unemployed for a loan)
# Create a dictionary to map labels to numbers
employment_map = {
    'Salaried': 4,'Business Owner': 3,
    'Self-Employed': 2,
    'Contract': 1,
    'Unemployed': 0
}


# Apply it to your dataframe
df_cleaned['employment_type'] = df_cleaned['employment_type'].map(employment_map)
print(df_cleaned.head())


#### Perform Startdard scaler on age
# Initialize the Scaler
scaler = StandardScaler()

# 1. Fit and transform the training data
# We use [[ ]] to keep it as a 2D array which the scaler expects
X_train['age'] = scaler.fit_transform(X_train[['age']])
X_test['age'] = scaler.transform(X_test[['age']])

# 2. Transform the test data 
# (Note: We only 'transform', NOT 'fit' on the test set)
X_test['age'] = scaler.transform(X_test[['age']])
print(X_test['age'])

print(df_cleaned.head())

print(df_cleaned.columns)

######################################################################################################
# Data Importing / Loading (cached)
######################################################################################################
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

######################################################################################################
# Model Training (cached resource)
######################################################################################################
@st.cache_resource
def train_model(df: pd.DataFrame):
    target = "approved"
    # We drop the target and the non-numeric name
    cols_to_drop = [target, "applicant_name"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target]

    # Categorical: Only columns that are still strings (e.g., gender, employment_type)
    # We EXCLUDE "city" because it is already split into city_karachi, etc.
    cat_cols = [c for c in ["gender", "employment_type", "bank"] if c in X.columns]
    
    # Numerical: Everything else (including the city_ Karachi, city_Lahore columns)
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    # Classifier
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    return clf, metrics, X.columns.tolist()



######################################################################################################
# Model Training (cached resource)
######################################################################################################
@st.cache_resource
def train_model(df:pd.DataFrame):
    #preprocessing
    target = "approved"

    drop_cols = [target]

    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")
        
    X = df.drop(columns =drop_cols)
    
    y =df[target]

    cat_cols = [c for c in ["gender","city","employment_type","bank",] if c in  X.columns]
    
    num_cols = [c for c in X.columns if c not in cat_cols]


    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(steps=[
        ("preprocess",preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy" : float(accuracy_score(y_test, y_pred)),
        # when we predict approved, how often is it correct?
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        # out of all truly approved, how many did we catch?
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        # balance between precision and recall
        "f1": float(f1_score(y_test,y_pred,zero_division=0)),
        # it shows TP, TN, FP, FN in a 2x2 tablw
        "confusion_matrix": confusion_matrix(y_test,y_pred).tolist()
    }
    return clf, metrics, X_train.columns.tolist()



######################################################################################################
# Sidebar (1) Load Dataset
######################################################################################################
st.sidebar.header("(1) Load Dataset")

csv_path = st.sidebar.text_input(
    "CSV Path",
    value = "loan_dataset.csv",
    help="Put the path to the dataset CSV. If you run from same folder, keep it as-is"
)

# Try loading the dataset
try:
    df = load_data(csv_path)

except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} rows")

######################################################################################################
# Sidebar (2) Train Model
######################################################################################################
st.sidebar.header("(2) Train Model")
train_now = st.sidebar.button("Train / Re-Train")

if train_now:
    st.cache_resource.clear()

clf, metrics, feature_order = train_model(df)

######################################################################################################
# MAIN LAYOUT
######################################################################################################

colA, colB = st.columns([1,1])

with colA:
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

with colB:
    st.subheader("Model Metrics (holdout test set)")

    st.write({
        "Accuracy": round(metrics["accuracy"],4),
        "Precision": round(metrics["precision"],4),
        "Recall": round(metrics["recall"],4),
        "F1": round(metrics["f1"],4),

    })

    cm = np.array(metrics["confusion_matrix"])
    st.write("Confusion Matrix (row: actual [0,1], cols: predicted [0,1])")

    st.dataframe(
        pd.DataFrame(cm, columns=["Pred 0","Pred 1"], index=["Actual 0","Actual 1"]),
        use_container_width=True
    )

st.divider()

######################################################################################################
# Trying A Prediction (UI Inputs)
######################################################################################################
# --- 3. UI LOGIC ---
st.title("🏦 Loan Process Application-PAKISTAN")

# Load Data
try:
    df = load_data("loan_dataset.csv")
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Train Model
clf, metrics, feature_order = train_model(df)

# --- 4. PREDICTION UI ---
st.subheader("Check your loan application eligibility")
c1, c2, c3 = st.columns(3)

with c1:
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Age", 18, 65, 30)
    # Get the list of cities by looking for columns that start with "city_"
    city_list = [c.replace("city_", "") for c in df.columns if c.startswith("city_")]
    selected_city = st.selectbox("City", sorted(city_list))

with c2:
    income = st.number_input("Monthly Income (PKR)", value=80000)
    c_score = st.slider("Credit Score", 300, 850, 650)
    emp = st.selectbox("Employment", df['employment_type'].unique() if 'employment_type' in df.columns else ["Salaried"])

with c3:
    loan_amt = st.number_input("Loan Amount (PKR)", value=500000)
    bank = st.selectbox("Bank", df['bank'].unique() if 'bank' in df.columns else ["HBL"])
    history = st.selectbox("Default History", [0, 1])

# --- 5. BUILDING THE INPUT ROW ---
# This part is tricky because we have to turn the single "City" choice back into 1s and 0s
input_dict = {
    "gender": gender,
    "age": age,
    "monthly_income_pkr": income,
    "credit_score": c_score,
    "employment_type": emp,
    "bank": bank,
    "default_history": history,
    "loan_amount_pkr": loan_amt,
    "loan_tenure_months": 12, # Default values for others
    "existing_loans": 0,
    "has_credit_card": 0
}

# Set all city columns to 0
for c in [col for col in df.columns if col.startswith("city_")]:
    input_dict[c] = 0

# Set the selected city column to 1
input_dict[f"city_{selected_city}"] = 1

# Convert to DataFrame and align columns
input_df = pd.DataFrame([input_dict])

# Final check: Make sure input_df has every column the model expects in the right order
input_df = input_df.reindex(columns=feature_order, fill_value=0)

if st.button("Predict Approval Status"):
    prediction = clf.predict(input_df)[0]
    probability = clf.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.success(f"✅ Approved! Probability: {probability:.2%}")
    else:
        st.error(f"❌ Rejected. Probability: {probability:.2%}")