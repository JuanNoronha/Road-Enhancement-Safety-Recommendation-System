from google.colab import files
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print("Model Accuracy:", accuracy)

# Identify the features (X) and the target (y)
X = data.drop(['Safety_Level'], axis=1)
y = data['Safety_Level']


# Separate categorical and numerical features
categorical_features = ['Million Plus Cities', 'Cause category', 'Cause Subcategory', 'Road Name']
numeric_features = [col for col in X.columns if col not in categorical_features]


city = st.selectbox('Select a city:', data['Million Plus Cities'].unique())
available_roads = data[data['Million Plus Cities'] == city]['Road Name'].unique()
selected_road = st.selectbox('Select a road name:', available_roads)
selected_data = data[(data['Million Plus Cities'] == city) & (data['Road Name'] == selected_road)]
if st.button("Show Safety Level"):
        if not selected_data.empty:
            safety_level = selected_data.iloc[0]['Safety_Level']
            st.write(f"Safety Level for {selected_road} in {city}: {safety_level}")
        else:
            st.write("No data found for the selected road in the city.")
