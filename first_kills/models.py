"""
Model definitions for first kills prediction.
"""

import pickle
import numpy as np
from typing import Optional, List, Dict
from sklearn.calibration import CalibratedClassifierCV
import warnings

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')


class FirstKillsModel:
    """
    Binary classifier for predicting first to N kills.
    
    Supports both first5_blue and first10_blue targets.
    """
    
    def __init__(self, target: str = 'first5_blue', random_state: int = 42, class_weights: Optional[List[float]] = None, custom_params: Optional[Dict] = None):
        """
        Initialize model.
        
        Args:
            target: Target label name ('first5_blue' or 'first10_blue')
            random_state: Random seed
            class_weights: Optional class weights [weight_red, weight_blue]
            custom_params: Optional dictionary of custom CatBoost parameters (overrides defaults)
        """
        self.target = target
        self.random_state = random_state
        self.feature_cols: Optional[List[str]] = None
        self.best_threshold: float = 0.5  # Default threshold, can be optimized
        
        # Initialize base classifier with target-specific config
        if CATBOOST_AVAILABLE:
            if custom_params is not None:
                # Use custom parameters (e.g., from hyperparameter tuning)
                params = custom_params.copy()
                params['random_seed'] = random_state
                params['class_weights'] = class_weights
                if 'verbose' not in params:
                    params['verbose'] = False
                self.model = CatBoostClassifier(**params)
            elif target == 'first10_blue':
                # Tuned config for first10 (from hyperparameter tuning)
                self.model = CatBoostClassifier(
                    iterations=600,
                    depth=4,
                    learning_rate=0.0188,
                    loss_function='Logloss',
                    eval_metric='Logloss',
                    l2_leaf_reg=1.91,
                    random_strength=1.48,
                    bagging_temperature=0.071,
                    subsample=0.91,
                    min_data_in_leaf=5,
                    grow_policy='SymmetricTree',
                    random_seed=random_state,
                    verbose=False,
                    early_stopping_rounds=10,
                    class_weights=class_weights
                )
            else:
                # Tuned config for first5 (from hyperparameter tuning)
                self.model = CatBoostClassifier(
                    iterations=350,
                    depth=4,
                    learning_rate=0.0265,
                    loss_function='Logloss',
                    eval_metric='Logloss',
                    l2_leaf_reg=4.91,
                    random_strength=1.64,
                    bagging_temperature=0.015,
                    subsample=0.70,
                    min_data_in_leaf=9,
                    grow_policy='SymmetricTree',
                    random_seed=random_state,
                    verbose=False,
                    early_stopping_rounds=35,
                    class_weights=class_weights
                )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                verbose=0
            )
        
        self.calibrator: Optional[CalibratedClassifierCV] = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_cols: Optional[List[str]] = None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_cols: List of feature column names
        """
        if feature_cols is not None:
            self.feature_cols = feature_cols
            X_train = X_train[feature_cols]
            if X_val is not None:
                X_val = X_val[feature_cols]
        
        # Handle missing values
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)
        
        # Train base model
        if CATBOOST_AVAILABLE and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Fit calibrator on validation set if available
        if X_val is not None and y_val is not None:
            try:
                # Use CalibratedClassifierCV with prefit (model already trained)
                # This will fit the calibrator on the validation set
                self.calibrator = CalibratedClassifierCV(
                    self.model,
                    method='isotonic',
                    cv='prefit'
                )
                self.calibrator.fit(X_val, y_val)
            except Exception as e:
                # If calibration fails, just use base model
                print(f"Warning: Calibration failed: {e}. Using uncalibrated model.")
                self.calibrator = None
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict probability that blue team is first to N kills.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of shape (n_samples, 2) with probabilities [P(red), P(blue)]
        """
        if self.feature_cols is not None:
            X = X[self.feature_cols]
        
        X = X.fillna(0)
        
        # Get base probabilities
        if self.calibrator is not None:
            proba = self.calibrator.predict_proba(X)
        else:
            proba = self.model.predict_proba(X)
        
        return proba
    
    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary class.
        
        Args:
            X: Feature DataFrame
            threshold: Probability threshold (defaults to self.best_threshold or 0.5)
            
        Returns:
            Array of predictions (0 = red, 1 = blue)
        """
        proba = self.predict_proba(X)
        thresh = threshold if threshold is not None else self.best_threshold
        return (proba[:, 1] >= thresh).astype(int)
    
    def set_threshold(self, threshold: float):
        """Set the optimal threshold for predictions."""
        self.best_threshold = threshold
    
    def save(self, model_path: str, calibrator_path: Optional[str] = None):
        """
        Save model to disk.
        
        Args:
            model_path: Path to save base model
            calibrator_path: Path to save calibrator (optional)
        """
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'target': self.target,
                'feature_cols': self.feature_cols,
                'random_state': self.random_state
            }, f)
        
        if self.calibrator is not None and calibrator_path is not None:
            with open(calibrator_path, 'wb') as f:
                pickle.dump(self.calibrator, f)
    
    @classmethod
    def load(cls, model_path: str, calibrator_path: Optional[str] = None):
        """
        Load model from disk.
        
        Args:
            model_path: Path to saved model
            calibrator_path: Path to saved calibrator (optional)
            
        Returns:
            Loaded FirstKillsModel instance
        """
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(target=data['target'], random_state=data['random_state'])
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        
        if calibrator_path is not None:
            try:
                with open(calibrator_path, 'rb') as f:
                    instance.calibrator = pickle.load(f)
            except FileNotFoundError:
                instance.calibrator = None
        
        return instance

