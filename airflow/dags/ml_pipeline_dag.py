"""
Main ML pipeline DAG for orchestrating the complete churn prediction workflow.
"""
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def load_data(**context):
    """Load or generate churn data."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags/../..')
    
    from src.data.data_loader import DataLoader
    from src.utils.logger import setup_logging
    
    setup_logging()
    loader = DataLoader()
    df = loader.load_data()
    train_df, val_df, test_df = loader.split_data(df)
    loader.save_split_data(train_df, val_df, test_df)
    
    context['ti'].xcom_push(key='data_loaded', value=True)
    return "Data loaded successfully"


def validate_data(**context):
    """Validate data quality."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags/../..')
    
    from src.data.data_loader import DataLoader
    from src.data.data_validator import DataValidator
    from src.utils.logger import setup_logging
    
    setup_logging()
    loader = DataLoader()
    df = loader.load_data()
    
    validator = DataValidator()
    results = validator.validate_data(df)
    
    if not results['is_valid']:
        raise ValueError("Data validation failed")
    
    context['ti'].xcom_push(key='data_validated', value=True)
    return "Data validated successfully"


def engineer_features(**context):
    """Engineer features for model training."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags/../..')
    
    from src.features.feature_engineering import FeatureEngineer
    from src.utils.helpers import load_pickle, save_pickle
    from src.utils.logger import setup_logging
    
    setup_logging()
    
    # Load split data
    train_df = load_pickle("data/processed/train.pkl")
    val_df = load_pickle("data/processed/val.pkl")
    test_df = load_pickle("data/processed/test.pkl")
    
    # Engineer features
    engineer = FeatureEngineer()
    X_train, y_train = engineer.prepare_features(train_df, fit=True)
    y_train_encoded = engineer.encode_target(y_train, fit=True)
    
    X_val, y_val = engineer.prepare_features(val_df, fit=False)
    y_val_encoded = engineer.encode_target(y_val, fit=False)
    
    X_test, y_test = engineer.prepare_features(test_df, fit=False)
    y_test_encoded = engineer.encode_target(y_test, fit=False)
    
    # Save processed data
    save_pickle((X_train, y_train_encoded), "data/processed/train_processed.pkl")
    save_pickle((X_val, y_val_encoded), "data/processed/val_processed.pkl")
    save_pickle((X_test, y_test_encoded), "data/processed/test_processed.pkl")
    
    # Save transformers
    engineer.save_transformers()
    
    context['ti'].xcom_push(key='features_engineered', value=True)
    return "Features engineered successfully"


def train_models(**context):
    """Train machine learning models."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags/../..')
    
    from src.models.train import ModelTrainer
    from src.utils.helpers import load_pickle, save_pickle
    from src.utils.logger import setup_logging
    
    setup_logging()
    
    # Load processed data
    X_train, y_train = load_pickle("data/processed/train_processed.pkl")
    X_val, y_val = load_pickle("data/processed/val_processed.pkl")
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Save results
    save_pickle(results, "models/training_results.pkl")
    
    context['ti'].xcom_push(key='models_trained', value=True)
    context['ti'].xcom_push(key='num_models', value=len(results))
    return f"Trained {len(results)} models successfully"


def evaluate_models(**context):
    """Evaluate and select best model."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags/../..')
    
    from src.models.train import ModelTrainer
    from src.models.evaluate import ModelEvaluator
    from src.utils.helpers import load_pickle, save_pickle
    from src.utils.logger import setup_logging
    
    setup_logging()
    
    # Load results and test data
    results = load_pickle("models/training_results.pkl")
    X_test, y_test = load_pickle("data/processed/test_processed.pkl")
    
    # Select best model
    trainer = ModelTrainer()
    best_algorithm, best_model, best_metrics = trainer.select_best_model(results)
    
    # Evaluate on test set
    evaluator = ModelEvaluator()
    test_metrics = evaluator.evaluate_model(best_model, X_test, y_test)
    
    # Save best model
    save_pickle(best_model, "models/best_model.pkl")
    save_pickle(test_metrics, "models/best_model_metrics.pkl")
    
    context['ti'].xcom_push(key='best_algorithm', value=best_algorithm)
    context['ti'].xcom_push(key='test_metrics', value=test_metrics)
    return f"Best model: {best_algorithm}"


def register_model(**context):
    """Register best model in MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags/../..')
    
    from src.models.train import ModelTrainer
    from src.utils.helpers import load_pickle
    from src.utils.logger import setup_logging
    
    setup_logging()
    
    # Load best model and metrics
    best_model = load_pickle("models/best_model.pkl")
    test_metrics = load_pickle("models/best_model_metrics.pkl")
    
    # Register model
    trainer = ModelTrainer()
    trainer.register_model(
        best_model,
        trainer.mlflow_config['model_name'],
        test_metrics
    )
    
    context['ti'].xcom_push(key='model_registered', value=True)
    return "Model registered successfully"


def deploy_model(**context):
    """Deploy model to production."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags/../..')
    
    from src.utils.logger import setup_logging
    
    setup_logging()
    
    # In a real deployment, this would:
    # 1. Copy model to production location
    # 2. Update API server configuration
    # 3. Restart API server
    # 4. Run smoke tests
    
    context['ti'].xcom_push(key='model_deployed', value=True)
    return "Model deployment completed"


# Create DAG
with DAG(
    'ml_pipeline_churn_prediction',
    default_args=default_args,
    description='End-to-end ML pipeline for churn prediction',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'churn', 'production'],
) as dag:
    
    # Task 1: Load data
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        provide_context=True,
    )
    
    # Task 2: Validate data
    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )
    
    # Task 3: Engineer features
    engineer_features_task = PythonOperator(
        task_id='engineer_features',
        python_callable=engineer_features,
        provide_context=True,
    )
    
    # Task 4: Train models
    train_models_task = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
        provide_context=True,
    )
    
    # Task 5: Evaluate models
    evaluate_models_task = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        provide_context=True,
    )
    
    # Task 6: Register model
    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
        provide_context=True,
    )
    
    # Task 7: Deploy model
    deploy_model_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        provide_context=True,
    )
    
    # Define task dependencies
    load_data_task >> validate_data_task >> engineer_features_task >> train_models_task
    train_models_task >> evaluate_models_task >> register_model_task >> deploy_model_task
