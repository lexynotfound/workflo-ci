import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


def preprocess_data(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_file)

    # Define features
    numerical_features = ['Expected salary (IDR)']
    categorical_features = ['Gender', 'Marital status', 'Highest formal of education',
                            'Current status', 'Experience']

    # Define pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Fit and transform
    processed_data = preprocessor.fit_transform(df)

    # Save
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
    np.save(os.path.join(output_dir, 'processed_data.npy'), processed_data)

    # Save feature names
    cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + cat_feature_names.tolist()

    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    # Save target
    desired_positions = df['Desired positions'].astype(str).fillna("").tolist()
    pd.DataFrame({'desired_positions': desired_positions}).to_csv(
        os.path.join(output_dir, 'target_data.csv'), index=False
    )

    print(f"[✓] Preprocessing completed. Files saved to {output_dir}")


if __name__ == "__main__":
    preprocess_data('../dataset/forminator-career-form-250124070425.csv', 'dataset/career_form_preprocessed')
