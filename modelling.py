import os
import numpy as np
import pandas as pd
import mlflow
import dagshub
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import matplotlib.pyplot as plt

# Set MLflow tracking URI (local)
# --- PILIH TARGET MLFLOW ---
TARGET = os.getenv("MLFLOW_TARGET", "dagshub")  # default ke dagshub

if TARGET == "dagshub":
    dagshub.init(repo_owner='lexynotfound', repo_name='mlops', mlflow=True)
    print("[MLflow] Logging to DagsHub")
elif TARGET == "local":
    mlflow.set_tracking_uri("http://localhost:8000")
    print("[MLflow] Logging to LOCALHOST")
else:
    raise ValueError("Unknown MLFLOW_TARGET. Use 'dagshub' or 'local'.")

# Set experiment
experiment_name = "candidate_recommendation_system"
mlflow.set_experiment(experiment_name)

def load_data(data_dir):
    """Load preprocessed data"""
    X = np.load(os.path.join(data_dir, 'processed_data.npy'))
    target_df = pd.read_csv(os.path.join(data_dir, 'target_data.csv'))
    return X, target_df


def train_model():
    """Train a candidate clustering model"""
    # Load preprocessed data
    data_dir = '../preprocessing/dataset/career_form_preprocessed'
    X, target_df = load_data(data_dir)

    # Start MLflow run
    with mlflow.start_run(run_name="candidate_clustering_model"):
        # Log parameters
        mlflow.log_param("data_path", data_dir)
        mlflow.log_param("data_shape", X.shape)

        # Split data for validation
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        # Train K-means clustering model (find optimal k)
        silhouette_scores = []
        k_values = range(2, 10)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_train)
            score = silhouette_score(X_train, cluster_labels)
            silhouette_scores.append(score)
            mlflow.log_metric(f"silhouette_score_k{k}", score)

        # Find optimal k
        optimal_k = k_values[np.argmax(silhouette_scores)]
        mlflow.log_param("optimal_k", optimal_k)

        # Train final model with optimal k
        final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_model.fit(X_train)

        # Evaluate on test set
        test_labels = final_model.predict(X_test)
        test_score = silhouette_score(X_test, test_labels)
        mlflow.log_metric("test_silhouette_score", test_score)

        # Log model
        mlflow.sklearn.log_model(final_model, "kmeans_model")

        # PERBAIKAN: Tangani nilai NaN dalam kolom desired_positions
        # Periksa dan tangani nilai NaN
        print(f"Total nilai NaN di desired_positions: {target_df['desired_positions'].isna().sum()}")

        # Opsi 1: Hapus baris dengan nilai NaN
        target_df_clean = target_df.copy()  # Gunakan copy untuk menghindari SettingWithCopyWarning
        target_df_clean = target_df_clean.dropna(subset=['desired_positions'])
        print(f"Jumlah baris setelah menghapus NaN: {len(target_df_clean)}")

        # Pastikan kolom memang berisi string - gunakan .loc untuk menghindari warning
        target_df_clean.loc[:, 'desired_positions'] = target_df_clean['desired_positions'].astype(str)

        # Process text data for job matching
        # Create TF-IDF vectorizer for job descriptions
        tfidf = TfidfVectorizer(stop_words='english')
        position_matrix = tfidf.fit_transform(target_df_clean['desired_positions'])

        # Log vectorizer
        mlflow.sklearn.log_model(tfidf, "tfidf_vectorizer")

        # Menggunakan cluster_centers dengan cara bermanfaat
        cluster_centers = final_model.cluster_centers_

        # 1. Simpan cluster_centers sebagai CSV dan log ke MLflow
        centers_df = pd.DataFrame(cluster_centers)
        centers_df.to_csv("cluster_centers.csv", index=False)
        mlflow.log_artifact("cluster_centers.csv")

        # 2. Hitung jarak antara posisi yang diinginkan dengan semua centroid
        # Kita bisa menggunakan ini untuk merekomendasikan cluster terbaik untuk posisi tertentu
        if position_matrix.shape[0] > 0:
            # Ambil posisi pertama sebagai contoh
            sample_position = position_matrix[0]

            # Hitung jarak ke semua centroid
            # Karena dimensi tidak sama, kita gunakan pendekatan sederhana
            # Ini adalah contoh pendekatan, dalam praktik nyata perlu disesuaikan
            distances = []
            for i, center in enumerate(cluster_centers):
                # Ambil beberapa fitur pertama (sesuaikan dengan feature space)
                if sample_position.shape[1] < center.shape[0]:
                    center_subset = center[:sample_position.shape[1]]
                    dist = np.linalg.norm(sample_position.toarray().flatten() - center_subset)
                else:
                    sample_subset = sample_position.toarray().flatten()[:center.shape[0]]
                    dist = np.linalg.norm(sample_subset - center)
                distances.append((i, dist))

            # Urutkan berdasarkan jarak (cluster terdekat = paling relevan)
            distances.sort(key=lambda x: x[1])

            # Log hasil ke MLflow
            for i, (cluster_idx, dist) in enumerate(distances[:3]):  # Log 3 cluster terdekat
                mlflow.log_metric(f"closest_cluster_{i + 1}", cluster_idx)
                mlflow.log_metric(f"closest_cluster_{i + 1}_distance", dist)

        # Log artifacts (helper files)
        joblib.dump(position_matrix, "position_matrix.pkl")
        mlflow.log_artifact("position_matrix.pkl")

        # Visualisasi cluster centers (bila dimensi memungkinkan)
        if cluster_centers.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, c='red', marker='X')
            # Tambahkan label cluster
            for i, (x, y) in enumerate(cluster_centers[:, :2]):
                plt.annotate(f"Cluster {i}", (x, y), textcoords="offset points",
                             xytext=(0, 10), ha='center')
            plt.title('Cluster Centers')
            plt.savefig('cluster_centers.png')
            mlflow.log_artifact('cluster_centers.png')

        # Log run info
        mlflow.log_param("model_type", "KMeans")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("n_clusters", optimal_k)
        mlflow.log_param("cluster_centers_shape", str(cluster_centers.shape))

        # Print run info
        print(f"Model trained with {X.shape[0]} samples and {X.shape[1]} features")
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Test silhouette score: {test_score:.4f}")
        print(f"Cluster centers shape: {cluster_centers.shape}")

        # Return model for further use
        return final_model, tfidf


if __name__ == "__main__":
    model, vectorizer = train_model()
    print("Model training completed and logged to MLflow")