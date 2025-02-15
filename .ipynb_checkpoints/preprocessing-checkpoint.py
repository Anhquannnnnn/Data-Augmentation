import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer




class DataPreprocessor:
    def __init__(self, 
                 numerical_features: list[str] = None,
                 categorical_features: list[str] = None,
                 scaling_method: str = 'standard',
                 categorical_encoding: str = 'onehot',
                 max_categories: int = 10):
        """
        Initialize the preprocessor with feature specifications
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
            scaling_method: 'standard', 'minmax', or 'robust'
            categorical_encoding: 'onehot' or 'label'
            max_categories: Maximum number of categories for one-hot encoding
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        self.max_categories = max_categories
        
        # Initialize transformers
        self.numerical_transformer = None
        self.categorical_transformer = None
        self.ordinal_encoders = {}
        self.column_transformer = None
        self.feature_names_out_ = None
    def fill_num(self,num: pd.DataFrame):
        null_map = num.isnull().sum()
        cols_contain_null = null_map.index[null_map >0 ]
        for col in cols_contain_null:
            num[col] = num[col].fillna(num[col].median())
        return num

        
    def _create_numerical_transformer(self):
        """Create transformer for numerical features"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
    def _create_categorical_transformer(self):
        """Create transformer for categorical features"""
        if self.categorical_encoding == 'onehot':
            return OneHotEncoder(sparse=False, handle_unknown='ignore', 
                               max_categories=self.max_categories)
        elif self.categorical_encoding == 'label':
            return LabelEncoder()
        else:
            raise ValueError(f"Unknown categorical encoding: {self.categorical_encoding}")
            
    def detect_feature_types(self, data: pd.DataFrame) -> None:
        """Automatically detect feature types if not specified"""
        if self.numerical_features is None and self.categorical_features is None:
            self.numerical_features = []
            self.categorical_features = []
            
            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    self.numerical_features.append(column)
                else:
                    self.categorical_features.append(column)
                    
            print(f"Detected {len(self.numerical_features)} numerical features")
            print(f"Detected {len(self.categorical_features)} categorical features")
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """Fit the preprocessor to the data"""
        # Detect feature types if not specified
        self.detect_feature_types(data)
        
        # Create transformers
        transformers = []
        
        # Numerical features
        if self.numerical_features:
            self.numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', self._create_numerical_transformer())
            ])
            transformers.append(('num', self.numerical_transformer, self.numerical_features))
        
        # Categorical features
        if self.categorical_features:
            self.categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', self._create_categorical_transformer())
            ])
            transformers.append(('cat', self.categorical_transformer, self.categorical_features))
        
        
        # Create and fit column transformer
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        self.column_transformer.fit(data)
        
        # Store feature names
        self.feature_names_out_ = (
            self.numerical_features +
            ([] if not self.categorical_features else 
             self.column_transformer.named_transformers_['cat']
             .named_steps['encoder'].get_feature_names(self.categorical_features).tolist()))
        
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform the data"""
        if self.column_transformer is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.column_transformer.transform(data)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform the data"""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, transformed_data: np.ndarray) -> pd.DataFrame:
        """Inverse transform the data back to original space"""
        if self.column_transformer is None:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        # Split the transformed data into parts for each transformer
        current_idx = 0
        parts = {}
        
        # Numerical features
        if self.numerical_features:
            n_numerical = len(self.numerical_features)
            parts['num'] = transformed_data[:, current_idx:current_idx + n_numerical]
            current_idx += n_numerical
        
        # Categorical features
        if self.categorical_features and self.categorical_encoding == 'onehot':
            n_categorical = len(self.column_transformer.named_transformers_['cat']
                              .named_steps['encoder'].get_feature_names(self.categorical_features))
            parts['cat'] = transformed_data[:, current_idx:current_idx + n_categorical]
            current_idx += n_categorical
        
        
        # Inverse transform each part
        inverse_transformed = {}
        
        # Numerical features
        if self.numerical_features:
            inverse_numerical = self.numerical_transformer.inverse_transform(parts['num'])
            for i, feature in enumerate(self.numerical_features):
                inverse_transformed[feature] = inverse_numerical[:, i]
        
        # Categorical features
        if self.categorical_features:
            if self.categorical_encoding == 'onehot':
                inverse_categorical = self.categorical_transformer.named_steps['encoder'].inverse_transform(parts['cat'])
                for i, feature in enumerate(self.categorical_features):
                    inverse_transformed[feature] = inverse_categorical[:, i]
        
        
        return pd.DataFrame(inverse_transformed)

# Example usage
def preprocess_data_for_tvae(data: pd.DataFrame, 
                            scaling_method: str = 'standard',
                            categorical_encoding: str = 'onehot',
                            max_categories: int = 10):
    """
    Preprocess data specifically for TVAE model
    
    Args:
        data: Input DataFrame
        scaling_method: Scaling method for numerical features
        categorical_encoding: Encoding method for categorical features
        max_categories: Maximum number of categories for one-hot encoding
        
    Returns:
        Tuple of (preprocessed_data, preprocessor)
    """
    # Create preprocessor
    preprocessor = DataPreprocessor(
        scaling_method=scaling_method,
        categorical_encoding=categorical_encoding,
        max_categories=max_categories
    )
    
    # Fit and transform data
    preprocessed_data = preprocessor.fit_transform(data)
    
    return preprocessed_data, preprocessor
