import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Union, Optional
import random
from preprocessing import in_out_to_list

def display_output(data, list_output):
    plt.figure()
    for i in list_output:
        plt.plot(in_out_to_list(data["OUTPUT"][i]), label = f"DATA = {data["DATA"][i]}, T= {data["T"][i]}")
        plt.legend()
    
    plt.grid()
    plt.show()


        
class DataAnalyzer:
    """
    A class for comprehensive data analysis including summary statistics,
    visualizations, and statistical tests.
    """
    
    def __init__(self, data: pd.DataFrame, numeric_columns,categorical_columns ):
        """
        Initialize the DataAnalyzer with a pandas DataFrame.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input data to analyze
        """
        self.data = data
        self.numeric_columns =numeric_columns
        self.categorical_columns = categorical_columns

        
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics for numeric columns.
        """
        summary_num = self.data[self.numeric_columns].describe()
        # Add additional statistics
        summary_num.loc['skewness'] = self.data[self.numeric_columns].skew()
        summary_num.loc['kurtosis'] = self.data[self.numeric_columns].kurtosis()

        summary_cat = self.data[self.categorical_columns].describe()
        return summary_num, summary_cat
    
    def analyze_distributions(self, columns: Optional[list[str]] = None) -> dict:
        """
        Perform normality tests and distribution analysis.
        
        Parameters:
        -----------
        columns : List[str], optional
            Specific columns to analyze. If None, analyzes all numeric columns.
        """
            
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



    def plot_distributions_cat(self, columns: Optional[list[str]] = None):
        """
        Create categorical distribution plots showing value counts for specified columns.
        """
        if columns is None:
            columns = self.categorical_columns
    
        # Handle empty columns case
        if not columns:
            print("No categorical columns to plot")
            return None
    
        # Calculate grid dimensions
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols  # More robust calculation
    
        # Create subplot grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = np.array(axes).flatten()  # Ensure axes is always an array
    
        # Plot each categorical distribution
        for idx, col in enumerate(columns):
            if col not in self.data.columns:
                continue
                
            # Create count plot with trimmed x-labels
            ax = axes[idx]
            sns.countplot(data=self.data, x=col, ax=ax, order=self.data[col].value_counts().index)
            ax.set_title(f'Distribution of {col}', pad=15)
            ax.set_xlabel('')  # Remove redundant x-axis label
            
            # Rotate labels and adjust layout
            ax.tick_params(axis='x', rotation=45 if self.data[col].nunique() > 5 else 0)
            ax.tick_params(axis='both', labelsize=9)
    
        # Hide empty subplots and adjust layout
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
    
        plt.tight_layout(pad=2.0)
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
