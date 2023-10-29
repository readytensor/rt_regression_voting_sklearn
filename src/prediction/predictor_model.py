import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

PREDICTOR_FILE_NAME = "Voting_regressor.joblib"


class Regressor:
    """A wrapper class for the Voting Regressor.

    This class provides a consistent interface that can be used with other
    regressor models.

    Attributes:
        model_name (str): Name of the regressor model.
    """

    model_name = "Voting Regressor"

    def __init__(self, **hyperparameters):
        """Construct a new Voting Regressor."""
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> VotingRegressor:
        """Build a new Voting regressor.

        Returns:
            VotingRegressor: Initialized Voting regressor.
        """
        base_learners = [
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("svr_rbf", SVR(kernel="rbf")),
            ("ridge", Ridge()),
            ("xgb", XGBRegressor(objective="reg:squarederror", random_state=42)),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
        ]
        model = VotingRegressor(
            estimators=base_learners,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the regressor to the training data.

        Args:
            train_inputs (pd.DataFrame): Training input data.
            train_targets (pd.Series): Training target data.
        """
        # Column-vector
        print(type(train_targets), train_targets.shape)
        if isinstance(train_targets, pd.DataFrame) and train_targets.shape[1] == 1:
            y = train_targets.values.ravel()
        else:
            y = train_targets
        self.model.fit(train_inputs, y)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.

        Args:
            inputs (pd.DataFrame): Input data for prediction.

        Returns:
            np.ndarray: Predicted regression targets.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the regressor and return the r-squared score.

        Args:
            test_inputs (pd.DataFrame): Test input data.
            test_targets (pd.Series): Test target data.

        Returns:
            float: R-squared score of the regressor.

        Raises:
            NotFittedError: If the model is not trained yet.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Directory path to save the model.

        Raises:
            NotFittedError: If the model is not trained yet.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Directory path from where to load the model.

        Returns:
            Regressor: Loaded regressor model.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        """String representation of the Regressor.

        Returns:
            str: Information about the regressor.
        """
        return f"Model name: {self.model_name}"


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_inputs (pd.DataFrame): The training data inputs.
        train_targets (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)
