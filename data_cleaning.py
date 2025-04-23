from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class DatasetCleaner:
    def __init__(self, df):
        self.df = df

    def dataset_info(df):
        """
        This function prints out some useful information about the dataset.

        The information displayed includes the dataset shape, a summary of the dataset
        information, the number of duplicate rows in the dataset, and the number of
        missing values in the dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            The pandas DataFrame object containing the dataset.

        Returns
        -------
        None
        """
        print("\n\n")
        print("Dataset Information:\n")
        print(f"Dataset shape: {df.shape}\n ------------------------------------- ")
        print(f"Dataset info:\n{df.info()}\n -------------------------------------")
        print(f"Number of Duplicates: {df.duplicated().sum()} \n -------------------------------------------")
        print(f"Number of Missing Values: {df.isnull().sum()}\n -------------------------------------------")
        
    def remove_features_with_many_nans(df, threshold=0.5):
        """
        Reads a dataset and removes features with more than a specified percentage of NaN values.

        Parameters:
        - threshold (float): Proportion of NaN values above which a feature is removed (default is 0.5).

        Returns:
        - pd.DataFrame: The cleaned dataset.
        """
        # Calculate the threshold count
        nan_threshold = threshold * len(df)

        # Identify features to be removed
        features_to_remove = df.columns[df.isnull().sum() > nan_threshold]

        # Remove features with more than the threshold count of NaN values
        cleaned_df = df.loc[:, df.isnull().sum() <= nan_threshold]

        if len(features_to_remove) == 0:
            print("No features removed due to excessive NaN values.")   
        else:
            print(f"Features removed due to excessive NaN values: {features_to_remove.tolist()}")

        return cleaned_df
            
    
    def handle_missing_values(df):
        """
        Loads a dataset and replaces NaN values based on the type of feature:
        - Continuous features: Replace NaN with the mean of the feature.
        - Categorical features: Replace NaN with the mode of the feature.

        Parameters:
        - df (pd.DataFrame): The dataset to be cleaned.
        Returns:
        - pd.DataFrame: The dataset with NaN values handled.
        """
        # Iterate over each column in the dataframe
        for column in df.columns:
            if df[column].dtype in ['float64', 'float32']:  # Continuous features
                mean_value = df[column].median()
                df[column] = df[column].fillna(mean_value)  # Direct assignment
            else:  # Categorical features
                mode_value = df[column].mode()[0]
                df[column] = df[column].fillna(mode_value)  # Direct assignment

        print("NaN values handled successfully.")
        return df
    
    def iqr_outliers(df):
        """
        Detects and removes outliers in a given column using the IQR method.

        Parameters:
        - df (pd.DataFrame): The dataset.

        Returns:
        - pd.DataFrame: The dataset with outliers removed.
        """
        outliers_count = 0
        for column in df.columns:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1  # Interquartile Range

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Count outliers
            column_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            outliers_count += column_outliers

            # Filter out outliers
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"Outliers removed: {outliers_count}")
        return df

    def dbscan_outliers(data, eps=0.5, min_samples=5, remove_outliers=True):
        """
        Detects and removes outliers using the DBSCAN clustering algorithm.

        Parameters:
        - data (pd.DataFrame): The dataset to process, must be numerical.
        - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        - remove_outliers (bool): If True, outliers (label = -1) will be removed.

        Returns:
        - pd.DataFrame: The dataset with outliers removed if remove_outliers is True.
        - list: The list of outlier indices.
        """
        # Step 1: Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Step 2: Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(data_scaled)
        print(labels)
        # Step 3: Identify outliers (label = -1)
        outlier_indices = [i for i, label in enumerate(labels) if label == -1]
        print(f"Number of Outliers identified: {len(outlier_indices)}")
        if remove_outliers:
            # Step 4: Remove outliers from the dataset
            data_cleaned = data.drop(index=outlier_indices)
            return data_cleaned, outlier_indices
        else:
            return data, outlier_indices

    @staticmethod
    def standardization (X_train, X_test = None ):
        """
        Standardize the continuous features of the training and test datasets.

        This function identifies the continuous features in the training set, scales them
        using the StandardScaler, and then applies the same transformation to the test set.

        Parameters:
        x_trian (pd.DataFrame): The training data.
        x_test (pd.DataFrame): The test data.

        Returns:
        X_train, X_test (tuple of pd.DataFrame): The standardized training and test data.
        """
        continuous_features = [col for col in X_train.columns if X_train[col].nunique() > 4]
    
        # Initialize the StandardScaler
        scaler = StandardScaler()
        
        # Scale continuous features in the training set
        X_train_scaled = X_train.copy()
        X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
        
        # If test data is provided, scale continuous features in the test set
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
        else:
            X_test_scaled = None

        return X_train_scaled, X_test_scaled