import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Union, Optional
import random



        
class DataAnalyzer:
    
    def __init__(self, data: pd.DataFrame, numeric_columns,categorical_columns ):
        self.data = data
        self.numeric_columns =numeric_columns
        self.categorical_columns = categorical_columns

        
    def get_summary_statistics(self) -> pd.DataFrame:
        summary_num = self.data[self.numeric_columns].describe()
        # Add additional statistics
        summary_num.loc['skewness'] = self.data[self.numeric_columns].skew()
        summary_num.loc['kurtosis'] = self.data[self.numeric_columns].kurtosis()

        summary_cat = self.data[self.categorical_columns].describe()
        return summary_num, summary_cat
    
    def analyze_distributions(self, columns: Optional[list[str]] = None) -> dict:   
        results = {}
        for col in columns:
            if col in self.numeric_columns:
                shapiro_stat, shapiro_p = stats.shapiro(self.data[col].dropna())
                ks_stat, ks_p = stats.kstest(
                    self.data[col].dropna(), 
                    'norm',
                    args=(self.data[col].mean(), self.data[col].std())
                )
                
                results[col] = {
                    'shapiro_test': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                    'ks_test': {'statistic': ks_stat, 'p_value': ks_p}
                }
        
        return results
    
    def chi_squared_test(self, col1: str, col2: str) -> dict:
        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'contingency_table': contingency_table,
            'expected_frequencies': pd.DataFrame(
                expected,
                index=contingency_table.index,
                columns=contingency_table.columns
            )
        }
    
    def correlation_analysis(self) -> pd.DataFrame:
        return self.data[self.numeric_columns].corr()
    
    def plot_distributions(self, columns: Optional[list[str]] = None):
        if columns is None:
            columns = self.numeric_columns
            
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if col in self.numeric_columns:
                sns.histplot(data=self.data, x=col, kde=True, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
                
        plt.tight_layout()
        return fig



    def plot_distributions_cat(self, columns: Optional[list[str]] = None):
        if columns is None:
            columns = self.categorical_columns

        if not columns:
            print("No categorical columns to plot")
            return None
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols  
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = np.array(axes).flatten() 
        for idx, col in enumerate(columns):
            if col not in self.data.columns:
                continue
            ax = axes[idx]
            sns.countplot(data=self.data, x=col, ax=ax, order=self.data[col].value_counts().index)
            ax.set_title(f'Distribution of {col}', pad=15)
            ax.set_xlabel('') 
            ax.tick_params(axis='x', rotation=45 if self.data[col].nunique() > 5 else 0)
            ax.tick_params(axis='both', labelsize=9)
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
    
        plt.tight_layout(pad=2.0)
        return fig
    
    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.correlation_analysis(),
            annot=True,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Correlation Heatmap')
        return plt.gcf()
    
    def plot_boxplots(self, columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.numeric_columns
            
        plt.figure(figsize=(12, 6))
        self.data[columns].boxplot()
        plt.xticks(rotation=45)
        plt.title('Boxplots of Numeric Variables')
        return plt.gcf()


    def generate_report(self) -> str:
        report = []
        report.append("Data Analysis Report")
        report.append("===================")
        report.append("\n1. Dataset Overview")
        report.append(f"Number of rows: {len(self.data)}")
        report.append(f"Number of columns: {len(self.data.columns)}")
        report.append(f"Missing values:\n{self.data.isnull().sum().to_string()}")
        report.append("\n2. Summary Statistics")
        report.append(self.get_summary_statistics().to_string())
        report.append("\n3. Distribution Analysis")
        dist_results = self.analyze_distributions()
        for col, tests in dist_results.items():
            report.append(f"\n{col}:")
            report.append(f"Shapiro-Wilk test p-value: {tests['shapiro_test']['p_value']:.4f}")
            report.append(f"Kolmogorov-Smirnov test p-value: {tests['ks_test']['p_value']:.4f}")
        
        return "\n".join(report)
    def plot_syn_data(self,param, noutput = 5):
        df = self.data[(self.data['CONFIG'] == param['CONFIG']) &(self.data['T'] == param['T']) & (self.data['EQUIPEMENT'] == param['EQUIPEMENT']) & (self.data['FREQUENCE'] == param['FREQUENCE'])]
        naff = min(df.shape[0], noutput)
        plt.figure(figsize= (18,12))
        for i in range(naff):
            plt.plot(df.iloc[i,9:].values, label = f"DATA = {round(df.iloc[i,0],2)}, DELTA = {round(df.iloc[i,4],2)},MESURE A = {round(df.iloc[i,5],2)}, MESURE A REF = {round(df.iloc[i,6],2)}, MESURE B = {round(df.iloc[i,7], 2)}")
        plt.grid()
        plt.legend()
        plt.title(f"CONFIG = {round(df.iloc[i,1], 2)}, T = {round(df.iloc[i,2],2)}, EQUIPEMENT = {round(df.iloc[i,3],2)},  FREQUENCE = {round(df.iloc[i,8],2)}")
        plt.show()

        