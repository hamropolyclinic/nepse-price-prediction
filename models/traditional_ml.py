"""
Traditional Machine Learning Models for NEPSE stock price prediction
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

logger = logging.getLogger(__name__)


class TraditionalMLModels:
    """Traditional Machine Learning models for stock price prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and scale data for training
        
        Args:
            X: Features
            y: Target values
            test_size: Test set size ratio
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['default'] = scaler
            
            logger.info(f"Data prepared: Train={len(X_train)}, Test={len(X_test)}")
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """
        Train Linear Regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models['linear_regression'] = model
            
            logger.info("Linear Regression model trained")
            return model
            
        except Exception as e:
            logger.error(f"Error training Linear Regression: {str(e)}")
            raise

    def train_ridge_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                               alpha: float = 1.0) -> Ridge:
        """
        Train Ridge Regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            alpha: Regularization strength
            
        Returns:
            Trained model
        """
        try:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            self.models['ridge'] = model
            
            logger.info(f"Ridge Regression model trained (alpha={alpha})")
            return model
            
        except Exception as e:
            logger.error(f"Error training Ridge Regression: {str(e)}")
            raise

    def train_lasso_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                               alpha: float = 0.1) -> Lasso:
        """
        Train Lasso Regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            alpha: Regularization strength
            
        Returns:
            Trained model
        """
        try:
            model = Lasso(alpha=alpha, max_iter=1000)
            model.fit(X_train, y_train)
            self.models['lasso'] = model
            
            logger.info(f"Lasso Regression model trained (alpha={alpha})")
            return model
            
        except Exception as e:
            logger.error(f"Error training Lasso Regression: {str(e)}")
            raise

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100, max_depth: int = 20) -> RandomForestRegressor:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Trained model
        """
        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models['random_forest'] = model
            
            logger.info(f"Random Forest model trained (trees={n_estimators})")
            return model
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            raise

    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               n_estimators: int = 100, learning_rate: float = 0.1) -> GradientBoostingRegressor:
        """
        Train Gradient Boosting model
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            
        Returns:
            Trained model
        """
        try:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            self.models['gradient_boosting'] = model
            
            logger.info(f"Gradient Boosting model trained (estimators={n_estimators})")
            return model
            
        except Exception as e:
            logger.error(f"Error training Gradient Boosting: {str(e)}")
            raise

    def train_adaboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       n_estimators: int = 100, learning_rate: float = 0.1) -> AdaBoostRegressor:
        """
        Train AdaBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            
        Returns:
            Trained model
        """
        try:
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
            model.fit(X_train, y_train)
            self.models['adaboost'] = model
            
            logger.info(f"AdaBoost model trained (estimators={n_estimators})")
            return model
            
        except Exception as e:
            logger.error(f"Error training AdaBoost: {str(e)}")
            raise

    def train_svr(self, X_train: np.ndarray, y_train: np.ndarray,
                  kernel: str = 'rbf', C: float = 100.0, gamma: str = 'scale') -> SVR:
        """
        Train Support Vector Regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
            
        Returns:
            Trained model
        """
        try:
            model = SVR(kernel=kernel, C=C, gamma=gamma)
            model.fit(X_train, y_train)
            self.models['svr'] = model
            
            logger.info(f"SVR model trained (kernel={kernel})")
            return model
            
        except Exception as e:
            logger.error(f"Error training SVR: {str(e)}")
            raise

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = 'model') -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'model': model_name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            logger.info(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                            cv: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            results = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            logger.info(f"Cross-validation: Mean R2={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with all trained models
        """
        try:
            self.train_linear_regression(X_train, y_train)
            self.train_ridge_regression(X_train, y_train, alpha=1.0)
            self.train_lasso_regression(X_train, y_train, alpha=0.1)
            self.train_random_forest(X_train, y_train, n_estimators=100)
            self.train_gradient_boosting(X_train, y_train, n_estimators=100)
            self.train_adaboost(X_train, y_train, n_estimators=100)
            self.train_svr(X_train, y_train, kernel='rbf', C=100.0)
            
            logger.info(f"All {len(self.models)} models trained successfully")
            return self.models
            
        except Exception as e:
            logger.error(f"Error training all models: {str(e)}")
            raise

    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with evaluation results
        """
        try:
            results = []
            
            for model_name, model in self.models.items():
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                results.append(metrics)
            
            results_df = pd.DataFrame(results)
            
            # Find best model
            best_idx = results_df['r2_score'].idxmax()
            self.best_model_name = results_df.loc[best_idx, 'model']
            self.best_model = self.models[self.best_model_name]
            
            logger.info(f"Best model: {self.best_model_name} (R2={results_df.loc[best_idx, 'r2_score']:.4f})")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error evaluating all models: {str(e)}")
            raise

    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using specified or best model
        
        Args:
            X: Features
            model_name: Name of model to use (None = best model)
            
        Returns:
            Predictions
        """
        try:
            if model_name is None:
                model = self.best_model
                model_name = self.best_model_name
            else:
                model = self.models[model_name]
            
            predictions = model.predict(X)
            logger.info(f"Predictions made using {model_name}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def feature_importance(self, model_name: str) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Args:
            model_name: Name of model
            
        Returns:
            DataFrame with feature importance
        """
        try:
            model = self.models[model_name]
            
            if not hasattr(model, 'feature_importances_'):
                logger.warning(f"{model_name} does not support feature importance")
                return None
            
            importance_df = pd.DataFrame({
                'feature_index': range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Feature importance retrieved for {model_name}")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise

    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        try:
            model = self.models[model_name]
            joblib.dump(model, filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def save_all_models(self, directory: str):
        """Save all trained models"""
        try:
            for model_name, model in self.models.items():
                filepath = f"{directory}/{model_name}_model.pkl"
                joblib.dump(model, filepath)
            logger.info(f"All models saved to {directory}")
        except Exception as e:
            logger.error(f"Error saving all models: {str(e)}")

    def load_all_models(self, directory: str):
        """Load all models from directory"""
        try:
            import os
            for filename in os.listdir(directory):
                if filename.endswith('_model.pkl'):
                    model_name = filename.replace('_model.pkl', '')
                    filepath = os.path.join(directory, filename)
                    model = joblib.load(filepath)
                    self.models[model_name] = model
            logger.info(f"Models loaded from {directory}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Traditional ML Models module initialized")
