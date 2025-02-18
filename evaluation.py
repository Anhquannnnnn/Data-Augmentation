import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
class EvaluateurDonneesSynthetiques:
    def __init__(self, donnees_originales : pd.DataFrame, donnees_synthetiques):
        """
        Initialise l'évaluateur avec les données originales et synthétiques.
        
        Args:
            donnees_originales: DataFrame ou array des données originales
            donnees_synthetiques: DataFrame ou array des données synthétiques
        """
        self.originales = donnees_originales
        self.synthetiques = donnees_synthetiques
        
    def statistiques_basiques(self):
        num_vars = self.originales.shape[1]
        cols = []
        for i in self.originales.columns:
            cols.append(f"{i}_original")
            cols.append(f"{i}_synthetique")
        stats_orig = np.array([np.mean(self.originales, axis=0), np.var(self.originales, axis=0), np.max(self.originales, axis=0), np.min(self.originales, axis=0)])
        stats_synth = np.array([np.mean(self.synthetiques, axis=0), np.var(self.synthetiques, axis=0), np.max(self.synthetiques, axis=0),np.min(self.synthetiques, axis=0)])
        stat_dict = {}
        for i in np.arange(0,2*num_vars,2):
            stat_dict[cols[i]] = stats_orig[:,i//2]
            stat_dict[cols[i+1]] = stats_synth[:,i//2]
            
            
        """Calcule les statistiques descriptives de base."""
        stat = pd.DataFrame( stat_dict, index = ['moyenne', 'variance', 'max', 'min'])

        """differences = {
            'diff_moyenne': np.abs(stats_orig['moyenne'] - stats_synth['moyenne']),
            'diff_variance': np.abs(stats_orig['variance'] - stats_synth['variance']),
            'diff_max': np.abs(stats_orig['max'] - stats_synth['max']),
            'diff_min': np.abs(stats_orig['min'] - stats_synth['min'])
        }"""
        stat.style.background_gradient(cmap='coolwarm')
        display(stat)
        
    
    def test_kolmogorov_smirnov(self):
        """Effectue le test de Kolmogorov-Smirnov pour chaque variable."""
        resultats = []
        for i in range(self.originales.shape[1]):
            statistic, p_value = stats.ks_2samp(self.originales[:, i], self.synthetiques[:, i])
            resultats.append({
                'variable': i,
                'statistique_ks': statistic,
                'p_value': p_value
            })
        return resultats
    
    def test_mann_whitney(self):
        """Effectue le test de Mann-Whitney pour chaque variable."""
        resultats = []
        for i in range(self.originales.shape[1]):
            statistic, p_value = stats.mannwhitneyu(self.originales[:, i], 
                                                  self.synthetiques[:, i],
                                                  alternative='two-sided')
            resultats.append({
                'variable': i,
                'statistique_mw': statistic,
                'p_value': p_value
            })
        return resultats
    
    def test_chi_carre(self, bins=10):
        """Effectue le test du Chi-carré pour chaque variable."""
        resultats = []
        for i in range(self.originales.shape[1]):
            # Créer des bins pour les données
            combined_range = (min(self.originales[:, i].min(), self.synthetiques[:, i].min()),
                            max(self.originales[:, i].max(), self.synthetiques[:, i].max()))
            
            hist_orig, _ = np.histogram(self.originales[:, i], bins=bins, range=combined_range)
            hist_synth, _ = np.histogram(self.synthetiques[:, i], bins=bins, range=combined_range)
            
            # Éviter la division par zéro
            hist_orig = hist_orig + 1
            hist_synth = hist_synth + 1
            
            statistic, p_value = stats.chisquare(hist_orig, hist_synth)
            resultats.append({
                'variable': i,
                'statistique_chi2': statistic,
                'p_value': p_value
            })
        return resultats
    
    def mmd_rbf(self, gamma=None):
        """Calcule la statistique MMD (Maximum Mean Discrepancy) avec noyau RBF."""
        if gamma is None:
            # Utiliser la médiane des distances comme paramètre gamma par défaut
            distances = []
            for i in range(min(1000, len(self.originales))):  # Limiter pour la performance
                for j in range(i + 1, min(1000, len(self.originales))):
                    distances.append(np.sum((self.originales[i] - self.originales[j]) ** 2))
            gamma = 1.0 / np.median(distances)
        
        K_XX = rbf_kernel(self.originales, self.originales, gamma)
        K_YY = rbf_kernel(self.synthetiques, self.synthetiques, gamma)
        K_XY = rbf_kernel(self.originales, self.synthetiques, gamma)
        
        mmd = (K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())
        return mmd
    
    def evaluer_tout(self):
        """Effectue tous les tests et retourne un rapport complet."""
        rapport = {
            'statistiques': self.statistiques_basiques(),
            'kolmogorov_smirnov': self.test_kolmogorov_smirnov(),
            'mann_whitney': self.test_mann_whitney(),
            'chi_carre': self.test_chi_carre(),
            'mmd_rbf': self.mmd_rbf()
        }
        return rapport


