import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

class EEGFeatureSelector:
    def __init__(self, quality_threshold=0.8, max_features=None, methods=None):
        self.quality_threshold = quality_threshold
        self.max_features = max_features
        
        if methods is None:
            self.methods = ['stability', 'mutual_info', 'forest', 'ensemble', 'clustering']
        else:
            self.methods = methods
            
        self.selected_features = {}
        self.feature_scores = {}
        self.feature_stability = {}
        self.final_features = None
        self.feature_importances = None
        
    def fit(self, X, y=None, cv=5):
        self.feature_names = X.columns.tolist()
        
        for method in self.methods:
            if method == 'stability':
                self._perform_stability_selection(X, y, cv)
            elif method == 'variance':
                self._select_by_variance_quality(X)
            elif method == 'correlation':
                self._select_by_correlation_structure(X)
            elif method == 'mutual_info' and y is not None:
                self._select_by_mutual_info_thresholding(X, y)
            elif method == 'forest' and y is not None:
                self._select_by_forest_importance(X, y)
            elif method == 'ensemble' and y is not None:
                self._select_by_ensemble_learning(X, y)
            elif method == 'clustering':
                self._select_by_feature_clustering(X)
            elif method == 'lasso' and y is not None:
                self._select_by_lasso_path(X, y, cv)
                
        self._aggregate_selections()
        return self
    
    def transform(self, X):
        if self.final_features is None:
            return X
        return X[self.final_features]
    
    def fit_transform(self, X, y=None, cv=5):
        self.fit(X, y, cv)
        return self.transform(X)
    
    def _perform_stability_selection(self, X, y, cv):
        if y is None:
            return
            
        n_samples, n_features = X.shape
        bootstrap_results = {}
        
        for i in range(cv):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X.iloc[indices], y[indices]
            
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=i*10),
                threshold='median'
            )
            selector.fit(X_boot, y_boot)
            
            selected = np.where(selector.get_support())[0]
            for feat_idx in selected:
                feat_name = self.feature_names[feat_idx]
                bootstrap_results[feat_name] = bootstrap_results.get(feat_name, 0) + 1
        
        stability_scores = pd.Series(bootstrap_results).sort_values(ascending=False) / cv
        
        self.feature_scores['stability'] = stability_scores
        self.selected_features['stability'] = stability_scores[
            stability_scores >= self.quality_threshold
        ].index.tolist()
        
    def _select_by_variance_quality(self, X):
        variances = X.var()
        
        # Use IQR to identify features with meaningful variance
        q1, q3 = variances.quantile([0.25, 0.75])
        iqr = q3 - q1
        threshold = q1 + iqr * 0.1  # Adaptive threshold based on data distribution
        
        quality_mask = variances >= threshold
        var_scores = variances / variances.max()  # Normalize scores
        
        self.feature_scores['variance'] = var_scores
        self.selected_features['variance'] = var_scores[quality_mask].index.tolist()
    
    def _select_by_correlation_structure(self, X):
        corr_matrix = X.corr().abs()
        
        # Calculate connectivity for each feature
        mean_corr = corr_matrix.mean()
        
        # Find clusters of highly correlated features
        clusters = {}
        processed = set()
        
        for feature in self.feature_names:
            if feature in processed:
                continue
                
            correlated = corr_matrix[feature][corr_matrix[feature] > 0.7].index.tolist()
            if len(correlated) > 1:  # At least one other correlated feature
                clusters[feature] = correlated
                processed.update(correlated)
        
        # Select representatives from each cluster
        selected = []
        for primary, group in clusters.items():
            variances = X[group].var()
            representative = variances.idxmax()  # Select feature with highest variance
            selected.append(representative)
            
        # Add unclustered features
        unclustered = [f for f in self.feature_names if f not in processed]
        selected.extend(unclustered)
        
        # Create score as inverse of average correlation
        corr_scores = 1 - mean_corr
        
        self.feature_scores['correlation'] = corr_scores
        self.selected_features['correlation'] = selected
    
    def _select_by_mutual_info_thresholding(self, X, y):
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.Series(mi_scores, index=self.feature_names)
        
        # Automatically determine threshold using percentile
        threshold = np.percentile(mi_scores, 100 * (1 - self.quality_threshold))
        
        self.feature_scores['mutual_info'] = mi_df
        self.selected_features['mutual_info'] = mi_df[mi_df >= threshold].index.tolist()
    
    def _select_by_forest_importance(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=self.feature_names)
        
        # Use cumulative importance for quality threshold
        sorted_imp = importances.sort_values(ascending=False)
        cumulative_imp = sorted_imp.cumsum()
        
        # Select features that contribute to quality_threshold of total importance
        cutoff_idx = (cumulative_imp >= self.quality_threshold).idxmax()
        selected = sorted_imp[:sorted_imp.index.get_loc(cutoff_idx) + 1].index.tolist()
        
        self.feature_scores['forest'] = importances
        self.selected_features['forest'] = selected
    
    def _select_by_ensemble_learning(self, X, y):
    # Combine multiple models for robust feature selection
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42))
        ]
        
        ensemble_scores = pd.Series(0.0, index=self.feature_names)
        
        for name, model in models:
            model.fit(X, y)
            
            if hasattr(model, 'feature_importances_'):
                scores = pd.Series(model.feature_importances_, index=self.feature_names)
            elif hasattr(model, 'coef_'):
                scores = pd.Series(np.abs(model.coef_[0]), index=self.feature_names)
            else:
                continue
                
            scores = scores / scores.sum()  # Normalize
            ensemble_scores += scores
            
        # In case no models had valid scores
        if ensemble_scores.sum() > 0:
            ensemble_scores = ensemble_scores / len(models)
        
        # Use cumulative importance with threshold
        sorted_scores = ensemble_scores.sort_values(ascending=False)
        cumulative = sorted_scores.cumsum()
        
        # Select features that contribute to quality_threshold of total importance
        if len(cumulative) > 0 and cumulative.max() > 0:
            cutoff_idx = (cumulative >= self.quality_threshold).idxmax()
            selected = sorted_scores[:sorted_scores.index.get_loc(cutoff_idx) + 1].index.tolist()
        else:
            selected = []
        
        self.feature_scores['ensemble'] = ensemble_scores
        self.selected_features['ensemble'] = selected
        
    def _select_by_feature_clustering(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters
        silhouette_scores = []
        min_clusters = max(2, int(len(self.feature_names) * 0.1))
        max_clusters = min(20, len(self.feature_names) // 2)
        
        cluster_range = range(min_clusters, max_clusters + 1)
        for n_clusters in cluster_range:
            fa = FeatureAgglomeration(n_clusters=n_clusters)
            cluster_labels = fa.fit(X_scaled.T).labels_
            
            if len(set(cluster_labels)) <= 1:
                continue
                
            score = silhouette_score(X_scaled.T, cluster_labels)
            silhouette_scores.append((n_clusters, score))
        
        if not silhouette_scores:
            self.feature_scores['clustering'] = pd.Series(1.0, index=self.feature_names)
            self.selected_features['clustering'] = self.feature_names
            return
            
        # Select best clustering
        best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        
        # Cluster features
        fa = FeatureAgglomeration(n_clusters=best_n_clusters)
        fa.fit(X_scaled.T)
        cluster_labels = fa.labels_
        
        # Select representative from each cluster
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(self.feature_names[i])
        
        selected = []
        cluster_scores = pd.Series(0.0, index=self.feature_names)
        
        for cluster_id, features in clusters.items():
            if len(features) == 1:
                selected.append(features[0])
                cluster_scores[features[0]] = 1.0
                continue
                
            # Choose feature with highest variance in each cluster
            variances = X[features].var()
            representative = variances.idxmax()
            selected.append(representative)
            
            # Score features within cluster by their variance
            norm_var = variances / variances.sum()
            for feat, var in norm_var.items():
                cluster_scores[feat] = var
        
        self.feature_scores['clustering'] = cluster_scores
        self.selected_features['clustering'] = selected
    
    def _select_by_lasso_path(self, X, y, cv):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find stability path
        lasso = LassoCV(cv=cv, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y)
        
        coefs = pd.Series(np.abs(lasso.coef_), index=self.feature_names)
        nonzero_coefs = coefs[coefs > 0]
        
        self.feature_scores['lasso'] = coefs
        self.selected_features['lasso'] = nonzero_coefs.index.tolist()
    
    def _aggregate_selections(self):
        if not self.selected_features:
            self.final_features = self.feature_names
            self.feature_importances = pd.Series(1.0, index=self.feature_names)
            return
        
        votes = Counter()
        
    
        for method, selected in self.selected_features.items():
            weight = 1.0
            if method in ['stability', 'forest', 'ensemble']:
                weight = 1.5 
                
            for feature in selected:
                votes[feature] += weight
    
 
        all_voted_features = list(votes.keys())
       
        n_methods = len(self.methods)
        consensus_scores = pd.Series({f: votes[f] / n_methods for f in all_voted_features})
        
        
        if self.max_features is not None:
            
            selected = consensus_scores.nlargest(self.max_features).index.tolist()
        else:
            
            consensus_threshold = self.quality_threshold * consensus_scores.max()
            selected = consensus_scores[consensus_scores >= consensus_threshold].index.tolist()
            
            
            if not selected:
                selected = consensus_scores.nlargest(max(5, len(self.feature_names) // 10)).index.tolist()
        
        self.final_features = selected
        self.feature_importances = consensus_scores
    
    def save_results(self, output_dir="feature_selection_results"):
        os.makedirs(output_dir, exist_ok=True)
        
        if self.final_features:
            pd.Series(self.final_features).to_csv(f"{output_dir}/selected_features.csv", header=False, index=False)
            
            if self.feature_importances is not None:
                self.feature_importances.sort_values(ascending=False).to_csv(f"{output_dir}/feature_importances.csv")
                
                plt.figure(figsize=(12, 8))
                top_n = min(30, len(self.feature_importances))
                top_features = self.feature_importances.nlargest(top_n)
                sns.barplot(x=top_features.values, y=top_features.index)
                plt.title("Selected Features by Consensus Score")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/selected_features.png")
                plt.close()
        
        for method, scores in self.feature_scores.items():
            scores.sort_values(ascending=False).to_csv(f"{output_dir}/{method}_scores.csv")
            
            plt.figure(figsize=(12, 8))
            top_n = min(30, len(scores))
            top_method_features = scores.nlargest(top_n)
            sns.barplot(x=top_method_features.values, y=top_method_features.index)
            plt.title(f"Top Features by {method}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{method}_scores.png")
            plt.close()
    
    def save_selected_dataframe(self, X, output_file="selected_features_data.csv"):
        if self.final_features:
            X_selected = X[self.final_features]
            X_selected.to_csv(output_file, index=False)
            return X_selected
        return X