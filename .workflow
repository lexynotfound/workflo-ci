
name: "Candidate Recommendation MLOps Pipeline"
description: "Complete MLOps pipeline for candidate recommendation system"


project:
  name: "candidate-recommendation"
  version: "1.0.0"
  author: "Kurnia Raihan Ardian"
  repository: "https://github.com/lexynotfound/mlOps"


mlflow:
  tracking_uri: "dagshub"
  experiment_name: "candidate_recommendation_system"
  artifact_location: "./mlruns"
  autolog: true


stages:
  preprocessing:
    script: "namadataset_preprocessing.py"
    parameters:
      input_data: "../dataset/forminator-career-form-250124070425.csv"
      output_dir: "../preprocessing/dataset/career_form_preprocessed"

  training:
    script: "modelling.py"
    depends_on: "preprocessing"
    parameters:
      data_path: "career_form_preprocessed"
      model_type: "kmeans"
      optuna_trials: 20

  evaluation:
    metrics:
      - "silhouette_score"
      - "calinski_harabasz_score"
      - "davies_bouldin_score"

  deployment:
    model_format: "sklearn"
    serving:
      port: 3000
      endpoint: "/predict"


environment:
  python_version: "3.10"
  conda_env: "conda.yaml"


resources:
  cpu: 2
  memory: "4GB"


monitoring:
  prometheus:
    enabled: true
    port: 8000
  grafana:
    enabled: true
    dashboard: "model_dashboard.json"


ci_cd:
  github_actions: true
  docker: true
  auto_deploy: false