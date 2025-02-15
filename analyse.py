import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Union, Optional

class DataAnalyzer:
    """
    A class for comprehensive data analysis including summary statistics,
    visualizations, and statistical tests.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataAnalyzer with a pandas DataFrame.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input data to analyze
        """
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns
        self.categorical_columns = data.select_dtypes(exclude=[np.number]).columns
        
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics for numeric columns.
        """
        summary = self.data[self.numeric_columns].describe()
        # Add additional statistics
        summary.loc['skewness'] = self.data[self.numeric_columns].skew()
        summary.loc['kurtosis'] = self.data[self.numeric_columns].kurtosis()
        return summary
    
    def analyze_distributions(self, columns: Optional[list[str]] = None) -> dict:
        """
        Perform normality tests and distribution analysis.
        
        Parameters:
        -----------
        columns : List[str], optional
            Specific columns to analyze. If None, analyzes all numeric columns.
        """
        if columns is None:
            columns = self.numeric_columns
            
        results = {}
        for col in columns:
            if col in self.numeric_columns:
                # Shapiro-Wilk test for normality
                shapiro_stat, shapiro_p = stats.shapiro(self.data[col].dropna())
                
                # Kolmogorov-Smirnov test
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
        """
        Perform Chi-squared test of independence between two categorical variables.
        """
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
        """
        Calculate correlation matrix for numeric columns.
        """
        return self.data[self.numeric_columns].corr()
    
    def plot_distributions(self, columns: Optional[list[str]] = None):
        """
        Create distribution plots for specified columns.
        """
        if columns is None:
            columns = self.numeric_columns
            
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if col in self.numeric_columns:
                # Histogram with KDE
                sns.histplot(data=self.data, x=col, kde=True, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
                
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self):
        """
        Create a correlation heatmap for numeric columns.
        """
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
        """
        Create boxplots for specified columns.
        """
        if columns is None:
            columns = self.numeric_columns
            
        plt.figure(figsize=(12, 6))
        self.data[columns].boxplot()
        plt.xticks(rotation=45)
        plt.title('Boxplots of Numeric Variables')
        return plt.gcf()

    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        """
        report = []
        report.append("Data Analysis Report")
        report.append("===================")
        
        # Basic information
        report.append("\n1. Dataset Overview")
        report.append(f"Number of rows: {len(self.data)}")
        report.append(f"Number of columns: {len(self.data.columns)}")
        report.append(f"Missing values:\n{self.data.isnull().sum().to_string()}")
        
        # Summary statistics
        report.append("\n2. Summary Statistics")
        report.append(self.get_summary_statistics().to_string())
        
        # Distribution analysis
        report.append("\n3. Distribution Analysis")
        dist_results = self.analyze_distributions()
        for col, tests in dist_results.items():
            report.append(f"\n{col}:")
            report.append(f"Shapiro-Wilk test p-value: {tests['shapiro_test']['p_value']:.4f}")
            report.append(f"Kolmogorov-Smirnov test p-value: {tests['ks_test']['p_value']:.4f}")
        
        return "\n".join(report)