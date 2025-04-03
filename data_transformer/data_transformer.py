from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo',
    ['column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'],
)

class DataTransformer(object):
    def __init__(self, max_clusters=10, weight_threshold=0.007):
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_continuous(self, data):
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold,
        )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type='continuous',
            transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components,
        )
    def _fit_discrete(self, data):
        column_name = data.columns[0]
        ohe = OneHotEncoder(sparse_output=False)
        ohe.fit(data)
        return ColumnTransformInfo(
            column_name=column_name,
            column_type='discrete',
            transform=ohe,
            output_info=[SpanInfo(len(ohe.categories_[0]), 'softmax')],
            output_dimensions=len(ohe.categories_[0])
        )
    def fit(self, raw_data, discrete_columns=()):
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)
            

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        transformed = gm.transform(data)
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output
        
    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        result = ohe.transform(data)
        if hasattr(result, 'toarray'):
            return result.toarray()
        else:
            return result 
        
    def _synchronous_transform(self, raw_data, column_transform_info_list):
        column_data_list = []
        for info in column_transform_info_list:
            if info.column_type == 'continuous': 
                column_data_list.append(self._transform_continuous(info, raw_data[[info.column_name]]))          
            else:
                column_data_list.append(self._transform_discrete(info, raw_data[[info.column_name]]))
                
        return  column_data_list    
    def _parallel_transform(self, raw_data, column_transform_info_list):
        processes = [
            delayed(self._transform_continuous if info.column_type == 'continuous' 
                    else self._transform_discrete)(info, raw_data[[info.column_name]])
            for info in column_transform_info_list
        ]
        return Parallel(n_jobs=-1)(processes)

    def transform(self, raw_data):
        if not isinstance(raw_data, pd.DataFrame):
            cols = [str(i) for i in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=cols)
        
        transform_method = self._synchronous_transform if raw_data.shape[0] < 500 else self._parallel_transform
        return np.concatenate(
            transform_method(raw_data, self._column_transform_info_list),
            axis=1
        ).astype(float)
    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes())).astype(float)
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value
        return gm.reverse_transform(data)
        
    def _inverse_transform_discrete(self, column_transform_info, column_data):
        return pd.Series(
            column_transform_info.transform.inverse_transform(column_data).flatten(),
            name=column_transform_info.column_name
        )
    def inverse_transform(self, data, sigmas=None):
        column_names = [info.column_name for info in self._column_transform_info_list]
        recovered_data = []
        st = 0          
        for info in self._column_transform_info_list:
            dim = info.output_dimensions
            if info.column_type == 'continuous':
                col_data = self._inverse_transform_continuous(info, data[:, st:st+dim], sigmas, st)
            else:
                col_data = self._inverse_transform_discrete(info, data[:, st:st+dim])
            recovered_data.append(col_data)
            st += dim
        
        result = pd.DataFrame(np.column_stack(recovered_data), 
                             columns=column_names).astype(self._column_raw_dtypes)
        return result if self.dataframe else result.values

    def convert_column_name_value_to_id(self, column_name, value):
        try:
            col_infos = self._column_transform_info_list
            target_idx = next(i for i, info in enumerate(col_infos) 
                            if info.column_name == column_name)
        except StopIteration:
            raise ValueError(f"Column '{column_name}' not found") from None
    
        discrete_count = sum(1 for info in col_infos[:target_idx] 
                          if info.column_type == 'discrete')
        data = pd.DataFrame([value], columns=[self._column_transform_info_list[target_idx].column_name])
        encoder = col_infos[target_idx].transform
        one_hot = encoder.transform(data)
    
        if not one_hot.any():
            raise ValueError(f"Value '{value}' not in column '{column_name}'")
    
        return {
            'discrete_column_id': discrete_count,
            'column_id': target_idx,
            'value_id': one_hot.argmax()
        }



































