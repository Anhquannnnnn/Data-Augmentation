import pandas as pd
import numpy as np
import pywt
from sklearn.decomposition import PCA
class DataPreprocessor:
    def __init__(self, categorical_cols=None, numerical_cols=None):

        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.shape_coefs = []
    def get_info(self, data):
        n_null = data.isnull().sum()
        d_type = data.dtypes
        n_modes = data.nunique()
        info = pd.DataFrame({
            'N null': n_null,
            'Data type': d_type,
            "N modalities": n_modes
        })
        return info
    
    def _string_to_list(self, in_out_str):
        if isinstance(in_out_str, str):
            list_str = in_out_str.split(", ")
            list_float = []
            for i in range(len(list_str)):
                a_convert = list_str[i]
                if i == 0:
                    a_convert = a_convert[1:]
                elif i == len(list_str)-1:
                    a_convert = a_convert[:-1]
                list_float.append(float(a_convert))
            return list_float
        else:
            return in_out_str
    
    def preprocess(self, data):
        param = data.iloc[:, 1:10]
        if self.categorical_cols:
            param[self.categorical_cols] = data[self.categorical_cols].astype('category')
        if 'INPUT' in data.columns:
            data['INPUT'] = data['INPUT'].apply(self._string_to_list)
        
        if 'OUTPUT' in data.columns:
            data['OUTPUT'] = data['OUTPUT'].apply(self._string_to_list)
            data = data.join(data['OUTPUT'].apply(pd.Series).add_prefix('OUTPUT_'))
        cleaned_data = data.dropna().reset_index(drop=True)
        cleaned_output = cleaned_data.iloc[:, 11:]
        
        return cleaned_data, cleaned_output


    def apply_pca(self, cleaned_data, cleaned_output, n_components=5, variance_threshold=0.95):

        pca_full = PCA()
        pca_full.fit(cleaned_output)
        
        if n_components is None:
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        explained_variance_pct = 100 * sum(pca_full.explained_variance_ratio_[:n_components])
        print(f"Information kept: {explained_variance_pct:.9f}%")
        
        pca = PCA(n_components)
        output_pca = pca.fit_transform(cleaned_output)
        
        pca_columns = [f"comp_{i}" for i in np.arange(1, n_components+1)]
        param_cols = cleaned_data.iloc[:, 1:10] if cleaned_data.shape[1] >= 10 else cleaned_data
        trainable_data = pd.concat([
            param_cols, 
            pd.DataFrame(data=output_pca, columns=pca_columns)
        ], axis=1)
        
        return trainable_data, pca
    def apply_wavelet_transform(self, cleaned_data, cleaned_output, wavelet='db4', level=2):
            wavelet_coeffs = []
            count = False
            for index, row in cleaned_output.iterrows():
                signal = row.values
                coeffs = pywt.wavedec(signal, wavelet, level=level)
                flat_coeffs = []
                for coeff_array in coeffs:
                    if not count:
                        self.shape_coefs.append(len(coeff_array))
                    flat_coeffs.extend(coeff_array)
                count = True
                wavelet_coeffs.append(flat_coeffs)

            df_all_coeffs = pd.DataFrame(wavelet_coeffs)
         
            coeff_types = ['approx'] + [f'detail_{i}' for i in range(1, level + 1)]
            col_idx = 0
            new_cols = []
            
            for i, coeff_type in enumerate(coeff_types):
                if i < len(coeffs): 
                    coeff_length = len(coeffs[i])
                    for j in range(coeff_length):
                        new_cols.append(f'{coeff_type}_{j}')
                    col_idx += coeff_length
            
            df_all_coeffs.columns = new_cols

            param_cols = cleaned_data.iloc[:, 1:10] if cleaned_data.shape[1] >= 10 else cleaned_data

            trainable_data = pd.concat([param_cols, df_all_coeffs], axis=1)
            
            return trainable_data
    def inverse_wavelet_transform(self, df_all_coeffs, original_shape, wavelet='db4', level=2):
        reconstructed_signals = []
        approx_len = self.shape_coefs[0]
        detail_lens = self.shape_coefs[1:]
        for index, row in df_all_coeffs.iterrows():
            coeffs_flat = row.values
            start_idx = 0
            coeff_arrays = []
            coeff_arrays.append(coeffs_flat[start_idx:start_idx + approx_len])
            start_idx += approx_len
            for length in detail_lens:
                coeff_arrays.append(coeffs_flat[start_idx:start_idx + length])
                start_idx += length
            
            reconstructed = pywt.waverec(coeff_arrays, wavelet)
            if len(reconstructed) > original_shape[1]:
                reconstructed = reconstructed[:original_shape[1]]            
            reconstructed_signals.append(reconstructed)
        df_reconstructed = pd.DataFrame(reconstructed_signals)
        df_reconstructed.columns = [f'OUTPUT_{i}' for i in range(df_reconstructed.shape[1])]
        
        return df_reconstructed