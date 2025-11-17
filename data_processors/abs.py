from pathlib import Path
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw_data"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed_data"

class AbstractDataProcessor(ABC):
    """
    Abstract base class for data processors.

    This class defines the basic structure for data preprocessing,
    including methods for loading data, processing, and splitting
    the data into training, validation, and testing sets.

    Attributes
    ----------
    data_path : str
        Path to the data file.
    Methods
    -------
    load_data()
        Abstract method to load data.
    process()
        Abstract method to process the data.
    split_data(preprocessed_data)
        Abstract method to split the preprocessed data into different sets.
    """

    def __init__(self,data_name, *args, **kwargs):
        """
        Initialize the AbstractDataPreprocessor.

        Parameters
        ----------
        data_path : str
            Path to the data file.
        """
        self.data_name = data_name
        self.task_name = kwargs.get("task_name", "default_task")
        self.allowed_tasks = kwargs.get("allowed_tasks", [])
        self.raw_data_path = kwargs.get("raw_data_path", RAW_DATA_PATH / self.data_name)
        self.processed_data_path = kwargs.get("processed_data_path", PROCESSED_DATA_PATH / self.task_name / f"{self.data_name}.npz")
        self.global_args = kwargs.get("global_args")
        self.global_args["dataset_name"] = self.data_name

    def process(self):
        """
        Process the loaded data.
        """

        if self._check_processed_data_exists():
            return dict(np.load(self.processed_data_path, allow_pickle=True))

        #if self.task_name not in self.allowed_tasks:
        #    raise ValueError(f"Task '{self.task_name}' is not allowed for dataset {self.data_name}. Allowed tasks are: {self.allowed_tasks}")

        if self.task_name == "prediction":
            processed_data = self.process_prediction()
        elif self.task_name == "classification":
            processed_data = self.process_classification()
        elif self.task_name == "imputation":
            processed_data = self.process_imputation()
        elif self.task_name == "anomaly_detection":
            processed_data = self.process_anomaly_detection()
        elif self.task_name == "regression":
            processed_data = self.process_regression()
        else:
            raise ValueError(f"Task '{self.task_name}' is not implemented for dataset {self.data_name}.")

        self._save_processed_data(processed_data)

        return processed_data



    def process_prediction(self):
        """
        Process the data for prediction tasks.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def process_classification(self):
        """
        Process the data for classification tasks.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def process_imputation(self):
        """
        Process the data for imputation tasks.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def process_anomaly_detection(self):
        """
        Process the data for anomaly detection tasks.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


    def process_regression(self):
        """
        Process the data for regression tasks.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


    def _check_processed_data_exists(self):
        """
        Check if the processed data exists.
        """
        return self.processed_data_path.exists()

    def _save_processed_data(self, data):
        """
        Save the processed data to a file.
        """
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_data_path, **data)
