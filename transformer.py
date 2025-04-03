import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
class CondVec:
    def __init__(self, data, categorical_columns, categorical_dims):
        self.categorical_columns = categorical_columns
        self.categorical_dims = categorical_dims
        self.n_categories = sum(categorical_dims.values())
        self.n_features = len(categorical_columns)
        self.data = data
        
    def sample_conditional_vector(self, batch_size):
        """Sample conditional vectors for training."""
        if self.n_features == 0:
            return None, None
        
        vec = np.zeros((batch_size, self.n_categories), dtype='float32')
        mask = np.zeros((batch_size, self.n_features), dtype='float32')
        
        for i in range(batch_size):
            # Choose a random discrete column
            feature_idx = np.random.choice(range(self.n_features))
            feature = self.categorical_columns[feature_idx]
            
            # Choose a random category from that column
            feature_dim = self.categorical_dims[feature]
            category_idx = np.random.choice(range(feature_dim))
            
            # Set mask and vec values
            mask[i, feature_idx] = 1
            vec[i, sum(list(self.categorical_dims.values())[:feature_idx]) + category_idx] = 1  
        
        return torch.from_numpy(vec), torch.from_numpy(mask)
    
    def generate_conditional_vector(self, conditions, batch_size):
        """Generate conditional vector based on conditions."""
        if self.n_features == 0:
            return None
            
        vec = np.zeros((batch_size, self.n_categories), dtype='float32')
        for feature, category in conditions.items():
            if feature in self.categorical_columns:
                feature_idx = self.categorical_columns.index(feature)
                category_idx = int(category)  # Assuming category is an index
                
                vec[:, sum(list(self.categorical_dims.values())[:feature_idx]) + category_idx] = 1
        return torch.from_numpy(vec)
        
        


class CTGANDataset(Dataset):
    def __init__(self, data, categorical_columns=None):
        self.data = data
        self.categorical_columns = categorical_columns if categorical_columns else []
        self.continuous_columns = [col for col in data.columns if col not in self.categorical_columns]
        
        # Create encoders for categorical columns and fit GMMs for continuous columns
        self.cond_vec = None
        self.transformer = DataTransformer(self.categorical_columns)
        self.transformer.fit(data)
        self.transformed_data = self.transformer.transform(data)
        
        if len(self.categorical_columns) > 0:
            self.cond_vec = CondVec(
                data, 
                categorical_columns=self.categorical_columns,
                categorical_dims=self.transformer.categorical_dims
            )
    def get_categorical_dims(self):
        if len(self.categorical_columns) > 0:
            return self.transformer.categorical_dims
        else:
            return 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transformed_data[idx]
    
    def sample(self, batch_size):
        """Sample data and conditional vectors for training."""
        # Sample data
        idx = np.random.choice(range(len(self)), batch_size)
        data = self.transformed_data[idx]
        
        # Sample conditional vectors if categorical columns exist
        if self.cond_vec:
            cond_vec, mask = self.cond_vec.sample_conditional_vector(batch_size)
            return data, cond_vec, mask
        
        return data, None, None
    def train_val_split(self, val_ratio = 0.2):
        return train_test_split(self.transformed_data, test_size=val_ratio)


class TransformedCTGANDataset(Dataset):
    def __init__(self, trans_data, categorical_columns=None, categorical_dims = 0 ):
        self.transformed_data = trans_data
        self.categorical_columns = categorical_columns if categorical_columns else []
        self.cond_vec = None
        if len(self.categorical_columns) > 0:
            self.cond_vec = CondVec(
                trans_data, 
                categorical_columns=self.categorical_columns,
                categorical_dims= categorical_dims
            )    
    def __len__(self):
        return len(self.transformed_data)
    def __getitem__(self, idx):
        return self.transformed_data[idx]
    def sample(self, batch_size):
        """Sample data and conditional vectors for training."""
        # Sample data
        idx = np.random.choice(range(len(self)), batch_size)
        data = self.transformed_data[idx]
        
        # Sample conditional vectors if categorical columns exist
        if self.cond_vec:
            cond_vec, mask = self.cond_vec.sample_conditional_vector(batch_size)
            return data, cond_vec, mask
        
        return data, None, None
        
class DataTransformer:
    """Transforms data between original space and CTGAN transformed space."""
    
    def __init__(self, categorical_columns, n_clusters_max = 10):
        self.categorical_columns = categorical_columns if categorical_columns else []
        self.categorical_dims = {}
        self.continuous_gmms = {}
        self.n_clusters_max = n_clusters_max  # Number of modes for GMM
        
    def fit(self, data):
        """Fit the data transformer."""
        # Process categorical columns
        for column in self.categorical_columns:
            categories = pd.Categorical(data[column]).categories
            self.categorical_dims[column] = len(categories)
            
        # Process continuous columns by fitting GMMs
        continuous_columns = [c for c in data.columns if c not in self.categorical_columns]
        
        for column in continuous_columns:
            col_data = data[column].values.reshape(-1, 1)
            best_gmm = None
            best_bic = np.inf
            # Test de différents nombres de clusters (1 à n_clusters_max)
            for n_components in range(1, self.n_clusters_max + 1):
                gmm = GaussianMixture(n_components=n_components)
                gmm.fit(col_data)
                bic = gmm.bic(col_data)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    
            self.continuous_gmms[column] = best_gmm
            
    def transform(self, data):
        """Transform data to CTGAN format."""
        result = []
        
        # Transform categorical columns to one-hot encoding
        for column in self.categorical_columns:
            one_hot = pd.get_dummies(data[column], prefix=column)
            result.append(one_hot.values)
            
        # Transform continuous columns with mode-specific normalization
        for column in data.columns:
            if column not in self.categorical_columns:
                col_data = data[column].values.reshape(-1, 1)
                gmm = self.continuous_gmms[column]
                
                # Get cluster assignments and probabilities
                clusters = gmm.predict(col_data)
                probs = gmm.predict_proba(col_data)
                
                # Normalize data based on Gaussian parameters
                normalized = np.zeros_like(col_data)
                for i in range(len(col_data)):
                    cluster = clusters[i]
                    mean = gmm.means_[cluster][0]
                    std = np.sqrt(gmm.covariances_[cluster][0][0])
                    normalized[i] = (col_data[i] - mean) / (4 * std)
                
                # Create encoded data: [normalized value, cluster_1_prob, ..., cluster_k_prob]
                encoded = np.zeros((len(col_data), gmm.n_components + 1))
                encoded[:, 0] = normalized.flatten()
                encoded[:, 1:] = probs
                
                result.append(encoded)
                
        # Combine all transformed columns
        if result:
            return np.concatenate(result, axis=1).astype('float32')
        return np.zeros((len(data), 0))
        
    def inverse_transform(self, transformed_data):
        """Convert transformed data back to original format."""
        # Create a DataFrame for the result
        result = pd.DataFrame()
        column_idx = 0
        
        # Inverse transform categorical columns
        for column in self.categorical_columns:
            dim = self.categorical_dims[column]
            one_hot = transformed_data[:, column_idx:column_idx + dim]
            
            # Convert one-hot back to categorical
            indices = np.argmax(one_hot, axis=1)
            # Récupérer les catégories originales
            try:
                categories = pd.Categorical(self.data[column]).categories
                result[column] = pd.Categorical.from_codes(indices, categories=categories)
            except:
                # Fallback en cas d'erreur
                result[column] = indices
            
            column_idx += dim
            
        # Inverse transform continuous columns
        for column in self.continuous_gmms:
            gmm = self.continuous_gmms[column]
            
            # Extract normalized value and cluster probabilities
            normalized = transformed_data[:, column_idx]
            probs = transformed_data[:, column_idx + 1:column_idx + 1 + gmm.n_components]
            
            # Convert back to original space
            cluster_idx = np.argmax(probs, axis=1)
            values = np.zeros(len(normalized))
            
            for i in range(len(normalized)):
                cluster = cluster_idx[i]
                mean = gmm.means_[cluster][0]
                std = np.sqrt(gmm.covariances_[cluster][0][0])
                values[i] = normalized[i] * (4 * std) + mean
                
            result[column] = values
            column_idx += gmm.n_components + 1
            
        return result