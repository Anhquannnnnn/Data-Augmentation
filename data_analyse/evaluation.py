import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class SyntheticDataEvaluator:
    """
    Classe pour évaluer la qualité des données synthétiques par rapport aux données réelles.
    Adaptée pour des données avec des paramètres catégoriels et des signaux de sortie.
    """
    
    def __init__(self, real_data, synthetic_data, categorical_cols, signal_cols):
        """
        Initialisation avec les données réelles et synthétiques.
        
        Args:
            real_data (pd.DataFrame): DataFrame contenant les données réelles
            synthetic_data (pd.DataFrame): DataFrame contenant les données synthétiques
            categorical_cols (list): Liste des colonnes catégorielles (ex: ["T", "Frequence", "Config", "Equipement"])
            signal_cols (list): Liste des colonnes de signal (ex: ["output0", "output1", ..., "output100"])
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.categorical_cols = categorical_cols
        self.signal_cols = signal_cols
        
        print(f"Données réelles: {self.real_data.shape}")
        print(f"Données synthétiques: {self.synthetic_data.shape}")
        
    def evaluate_all(self):
        """Exécute toutes les évaluations disponibles"""
        print("\n=== ÉVALUATION COMPLÈTE DES DONNÉES SYNTHÉTIQUES ===\n")
        
        print("1. Statistiques de base:")
        self.basic_statistics()
        
        print("\n2. Distribution des paramètres catégoriels:")
        self.categorical_distribution()
        
        print("\n3. Analyse des signaux:")
        self.signal_analysis()
        
        print("\n4. Évaluation de la cohérence paramètres-signaux:")
        self.parameter_signal_coherence()
        
        print("\n5. Visualisation des distributions:")
        self.visualize_distributions()
        
        print("\n6. Évaluation par apprentissage supervisé:")
        self.supervised_evaluation()
        
    def basic_statistics(self):
        """Statistiques descriptives de base sur les signaux"""
        real_signal = self.real_data[self.signal_cols]
        synth_signal = self.synthetic_data[self.signal_cols]
        
        # Statistiques globales sur tous les signaux
        real_stats = {
            'moyenne': real_signal.values.mean(),
            'médiane': np.median(real_signal.values),
            'écart-type': real_signal.values.std(),
            'min': real_signal.values.min(),
            'max': real_signal.values.max(),
        }
        
        synth_stats = {
            'moyenne': synth_signal.values.mean(),
            'médiane': np.median(synth_signal.values),
            'écart-type': synth_signal.values.std(),
            'min': synth_signal.values.min(),
            'max': synth_signal.values.max(),
        }
        
        stats_df = pd.DataFrame({
            'Réel': real_stats,
            'Synthétique': synth_stats,
            'Différence (%)': {k: (synth_stats[k] - real_stats[k])/real_stats[k]*100 
                              for k in real_stats if real_stats[k] != 0}
        })
        
        print(stats_df)
        
        # Test de Kolmogorov-Smirnov
        ks_stat, ks_pval = stats.ks_2samp(
            real_signal.values.flatten(), 
            synth_signal.values.flatten()
        )
        print(f"\nTest Kolmogorov-Smirnov (global): stat={ks_stat:.4f}, p-value={ks_pval:.4f}")
        if ks_pval < 0.05:
            print("  → Les distributions sont significativement différentes (p < 0.05)")
        else:
            print("  → Pas de différence significative détectée entre les distributions")
            
    def categorical_distribution(self):
        """Analyse des distributions des colonnes catégorielles"""
        for col in self.categorical_cols:
            # Distribution de chaque catégorie
            real_dist = self.real_data[col].value_counts(normalize=True)
            synth_dist = self.synthetic_data[col].value_counts(normalize=True)
            
            # Alignement des index pour comparaison
            all_values = sorted(set(real_dist.index) | set(synth_dist.index))
            dist_df = pd.DataFrame(0, index=all_values, columns=['Réel', 'Synthétique'])
            
            for val in real_dist.index:
                dist_df.loc[val, 'Réel'] = real_dist[val]
            for val in synth_dist.index:
                dist_df.loc[val, 'Synthétique'] = synth_dist[val]
                
            print(f"\nDistribution de {col}:")
            print(dist_df)
            
            # Test Chi-carré
            if len(all_values) > 1:  # Nécessaire d'avoir au moins 2 catégories
                # Adapter pour les comptes réels (pas les proportions)
                real_counts = (dist_df['Réel'] * len(self.real_data)).astype(int)
                synth_counts = (dist_df['Synthétique'] * len(self.synthetic_data)).astype(int)
                
                # Éviter les valeurs nulles dans le test chi-carré
                real_counts = real_counts.replace(0, 1)
                synth_counts = synth_counts.replace(0, 1)
                
                chi2_stat, chi2_pval = stats.chisquare(synth_counts, f_exp=real_counts)
                print(f"  Test Chi-carré: stat={chi2_stat:.4f}, p-value={chi2_pval:.4f}")
                if chi2_pval < 0.05:
                    print("  → Différence significative dans la distribution (p < 0.05)")
                else:
                    print("  → Pas de différence significative détectée")
    
    def signal_analysis(self):
        """Analyse des signaux (par groupe de paramètres si spécifié)"""
        # Analyse globale des signaux
        real_signals = self.real_data[self.signal_cols].values
        synth_signals = self.synthetic_data[self.signal_cols].values
        
        # 1. Analyse de l'énergie moyenne des signaux
        real_energy = np.mean(np.sum(real_signals**2, axis=1))
        synth_energy = np.mean(np.sum(synth_signals**2, axis=1))
        energy_diff_pct = (synth_energy - real_energy) / real_energy * 100
        
        print(f"Énergie moyenne du signal:")
        print(f"  Réel: {real_energy:.4f}")
        print(f"  Synthétique: {synth_energy:.4f}")
        print(f"  Différence: {energy_diff_pct:.2f}%")
        
        # 2. Analyse spectrale (moyenne des FFT)
        real_fft = np.abs(fft(real_signals, axis=1))
        synth_fft = np.abs(fft(synth_signals, axis=1))
        
        real_fft_mean = np.mean(real_fft, axis=0)
        synth_fft_mean = np.mean(synth_fft, axis=0)
        
        # Distance euclidienne entre les spectres moyens
        fft_distance = euclidean(real_fft_mean, synth_fft_mean)
        fft_similarity = 1 / (1 + fft_distance)  # Normalisation en mesure de similarité [0-1]
        
        print(f"\nSimilarité spectrale (FFT): {fft_similarity:.4f} (1=identique, 0=très différent)")
        
        # 3. Autocorrélation moyenne (pour les premiers décalages)
        n_lags = min(20, len(self.signal_cols) // 2)
        
        real_autocorr = np.zeros((len(self.real_data), n_lags))
        synth_autocorr = np.zeros((len(self.synthetic_data), n_lags))
        
        # Calculer l'autocorrélation pour chaque signal
        for i in range(len(self.real_data)):
            signal = self.real_data[self.signal_cols].iloc[i].values
            real_autocorr[i] = [np.corrcoef(signal[:-lag], signal[lag:])[0,1] 
                               if lag > 0 else 1.0 for lag in range(n_lags)]
            
        for i in range(len(self.synthetic_data)):
            signal = self.synthetic_data[self.signal_cols].iloc[i].values
            synth_autocorr[i] = [np.corrcoef(signal[:-lag], signal[lag:])[0,1] 
                                if lag > 0 else 1.0 for lag in range(n_lags)]
            
        real_autocorr_mean = np.mean(real_autocorr, axis=0)
        synth_autocorr_mean = np.mean(synth_autocorr, axis=0)
        
        autocorr_distance = euclidean(real_autocorr_mean, synth_autocorr_mean)
        autocorr_similarity = 1 / (1 + autocorr_distance)
        
        print(f"Similarité d'autocorrélation: {autocorr_similarity:.4f} (1=identique, 0=très différent)")
        
    def parameter_signal_coherence(self):
        """Évalue la cohérence entre les paramètres et les signaux"""
        # Pour chaque combinaison de paramètres, comparer les signaux réels et synthétiques
        # Nous allons sélectionner quelques combinaisons fréquentes pour l'analyse
        
        # 1. Trouver les combinaisons présentes dans les deux ensembles
        real_params = self.real_data[self.categorical_cols].drop_duplicates()
        synth_params = self.synthetic_data[self.categorical_cols].drop_duplicates()
        
        # Trouver les combinaisons communes
        real_params_tuples = [tuple(x) for x in real_params.values]
        synth_params_tuples = [tuple(x) for x in synth_params.values]
        common_params = set(real_params_tuples) & set(synth_params_tuples)
        
        print(f"Nombre total de combinaisons de paramètres:")
        print(f"  Réel: {len(real_params_tuples)}")
        print(f"  Synthétique: {len(synth_params_tuples)}")
        print(f"  Commun: {len(common_params)}")
        
        # Analyser quelques combinaisons communes (max 5)
        n_samples = min(5, len(common_params))
        sample_params = list(common_params)[:n_samples]
        
        for param_tuple in sample_params:
            param_dict = {col: val for col, val in zip(self.categorical_cols, param_tuple)}
            
            # Filtre pour trouver tous les signaux avec cette combinaison de paramètres
            real_filter = True
            synth_filter = True
            
            for col, val in param_dict.items():
                real_filter &= (self.real_data[col] == val)
                synth_filter &= (self.synthetic_data[col] == val)
                
            real_subset = self.real_data[real_filter]
            synth_subset = self.synthetic_data[synth_filter]
            
            if len(real_subset) == 0 or len(synth_subset) == 0:
                continue
                
            print(f"\nAnalyse pour paramètres: {param_dict}")
            print(f"  Échantillons réels: {len(real_subset)}")
            print(f"  Échantillons synthétiques: {len(synth_subset)}")
            
            # Comparer les moyennes des signaux pour cette combinaison
            real_signal_mean = real_subset[self.signal_cols].mean().values
            synth_signal_mean = synth_subset[self.signal_cols].mean().values
            
            signal_distance = euclidean(real_signal_mean, synth_signal_mean)
            signal_similarity = 1 / (1 + signal_distance / len(self.signal_cols))
            
            print(f"  Similarité du signal moyen: {signal_similarity:.4f}")
            
            # Comparer les écarts-types des signaux pour cette combinaison
            real_signal_std = real_subset[self.signal_cols].std().values
            synth_signal_std = synth_subset[self.signal_cols].std().values
            
            std_distance = euclidean(real_signal_std, synth_signal_std)
            std_similarity = 1 / (1 + std_distance / len(self.signal_cols))
            
            print(f"  Similarité de la variabilité: {std_similarity:.4f}")
            
    def visualize_distributions(self):
        """Visualiser les distributions des signaux et paramètres"""
        # 1. Distribution globale des valeurs de signal (histogramme)
        plt.figure(figsize=(10, 6))
        
        real_values = self.real_data[self.signal_cols].values.flatten()
        synth_values = self.synthetic_data[self.signal_cols].values.flatten()
        
        plt.hist(real_values, bins=50, alpha=0.5, label='Réel', density=True)
        plt.hist(synth_values, bins=50, alpha=0.5, label='Synthétique', density=True)
        
        plt.title('Distribution des valeurs de signal')
        plt.xlabel('Valeur')
        plt.ylabel('Densité')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 2. Analyse en composantes principales des signaux
        pca = PCA(n_components=2)
        
        # Concaténer réel et synthétique pour PCA commune
        all_signals = np.vstack([
            self.real_data[self.signal_cols].values,
            self.synthetic_data[self.signal_cols].values
        ])
        
        pca_result = pca.fit_transform(all_signals)
        
        # Séparer les résultats
        n_real = len(self.real_data)
        real_pca = pca_result[:n_real]
        synth_pca = pca_result[n_real:]
        
        # Visualiser PCA
        plt.figure(figsize=(10, 6))
        plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Réel', s=10)
        plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label='Synthétique', s=10)
        
        plt.title(f'PCA des signaux (variance expliquée: {pca.explained_variance_ratio_.sum():.2f})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 3. Visualiser quelques signaux moyens
        # Sélectionner une combinaison de paramètres présente dans les deux ensembles
        common_combo = None
        for _, real_row in self.real_data[self.categorical_cols].drop_duplicates().iterrows():
            match = True
            for col in self.categorical_cols:
                match &= (self.synthetic_data[col] == real_row[col]).any()
            
            if match:
                common_combo = real_row
                break
                
        if common_combo is not None:
            # Filtrer les données
            real_filter = True
            synth_filter = True
            
            for col in self.categorical_cols:
                real_filter &= (self.real_data[col] == common_combo[col])
                synth_filter &= (self.synthetic_data[col] == common_combo[col])
                
            real_subset = self.real_data[real_filter]
            synth_subset = self.synthetic_data[synth_filter]
            
            # Afficher les signaux moyens
            if len(real_subset) > 0 and len(synth_subset) > 0:
                plt.figure(figsize=(12, 6))
                
                real_mean = real_subset[self.signal_cols].mean().values
                synth_mean = synth_subset[self.signal_cols].mean().values
                
                plt.plot(real_mean, label='Réel (moyenne)', linewidth=2)
                plt.plot(synth_mean, label='Synthétique (moyenne)', linewidth=2, linestyle='--')
                
                # Afficher la plage de ±1 écart-type
                real_std = real_subset[self.signal_cols].std().values
                synth_std = synth_subset[self.signal_cols].std().values
                
                x = np.arange(len(real_mean))
                plt.fill_between(x, real_mean - real_std, real_mean + real_std, alpha=0.2, color='blue')
                plt.fill_between(x, synth_mean - synth_std, synth_mean + synth_std, alpha=0.2, color='orange')
                
                plt.title(f'Signal moyen pour: {dict(common_combo)}')
                plt.xlabel('Temps')
                plt.ylabel('Valeur')
                plt.legend()
                plt.tight_layout()
                plt.show()
        
    def supervised_evaluation(self):
        """Évaluation par apprentissage supervisé"""
        # 1. Classification réel vs synthétique
        all_data = pd.concat([
            self.real_data[self.signal_cols].assign(is_real=1),
            self.synthetic_data[self.signal_cols].assign(is_real=0)
        ])
        
        X = all_data[self.signal_cols].values
        y = all_data['is_real'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Précision classification réel vs synthétique: {accuracy:.4f}")
        print(f"Note: Une précision proche de 0.5 est idéale (le classifieur ne peut pas distinguer)")
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        
        print("\nMatrice de confusion (normalisée):")
        cm_df = pd.DataFrame(cm_norm, 
                           index=['Réel', 'Synthétique'], 
                           columns=['Prédit Réel', 'Prédit Synthétique'])
        print(cm_df)
        
        # 2. Capacité prédictive des signaux
        # Sélectionner une colonne catégorielle pour la prédiction
        if len(self.categorical_cols) > 0:
            target_col = self.categorical_cols[0]  # Première colonne catégorielle
            
            print(f"\nTest de prédiction de {target_col} à partir du signal:")
            
            # Transformer cible catégorielle en numérique
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            
            # Entraînement sur réel, test sur synthétique
            real_X = self.real_data[self.signal_cols].values
            real_y = le.fit_transform(self.real_data[target_col])
            
            synth_X = self.synthetic_data[self.signal_cols].values
            synth_y = le.transform(self.synthetic_data[target_col])
            
            # Subdiviser les données réelles
            real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(
                real_X, real_y, test_size=0.3, random_state=42)
            
            # Entraîner sur réel
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(real_X_train, real_y_train)
            
            # Tester sur réel
            real_test_acc = accuracy_score(real_y_test, clf.predict(real_X_test))
            # Tester sur synthétique
            synth_test_acc = accuracy_score(synth_y, clf.predict(synth_X))
            
            print(f"  Réel → Réel: {real_test_acc:.4f}")
            print(f"  Réel → Synthétique: {synth_test_acc:.4f}")
            
            # Entraînement sur synthétique, test sur réel
            synth_X_train, synth_X_test, synth_y_train, synth_y_test = train_test_split(
                synth_X, synth_y, test_size=0.3, random_state=42)
            
            # Entraîner sur synthétique
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(synth_X_train, synth_y_train)
            
            # Tester sur synthétique
            synth_test_acc = accuracy_score(synth_y_test, clf.predict(synth_X_test))
            # Tester sur réel
            real_test_acc = accuracy_score(real_y, clf.predict(real_X))
            
            print(f"  Synthétique → Synthétique: {synth_test_acc:.4f}")
            print(f"  Synthétique → Réel: {real_test_acc:.4f}")
