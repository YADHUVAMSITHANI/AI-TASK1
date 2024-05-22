import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
data = fetch_openml(name='adult', version=2, as_frame=True)
df = data.frame

# Display the first few rows of the dataset
print(df.head())

# Data Cleaning
# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Imputation transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Separate features and target
X = df.drop('class', axis=1)  # 'class' is the target column in the 'adult' dataset
y = df['class']

# Encode the target variable if it is categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Fit the preprocessor on the data
preprocessor.fit(X)

# Transform the data
X_transformed = preprocessor.transform(X)

# Extract the fitted OneHotEncoder instance and get feature names
onehot_encoder = preprocessor.named_transformers_['cat']['onehot']
onehot_encoder.fit(df[categorical_features])  # Ensure OneHotEncoder is fitted
onehot_columns = onehot_encoder.get_feature_names_out(categorical_features)

# Combine numerical and one-hot encoded feature names
all_columns = numeric_features.tolist() + onehot_columns.tolist()

# Convert to DataFrame to keep column names
X_transformed_df = pd.DataFrame(X_transformed, columns=all_columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed_df, y, test_size=0.2, random_state=42)

# Output the processed data (optional)
print("X_train:\n", X_train.head())
print("X_test:\n", X_test.head())
print("y_train:\n", y_train[:5])
print("y_test:\n", y_test[:5])

# The processed data is now ready for AI model training
