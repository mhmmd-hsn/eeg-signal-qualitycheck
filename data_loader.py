import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import torch
from connections import GMatrixCalculator
from torch.utils.data import Dataset
from signal_evaluator import EEGQualityAnalyzer

class EEGDataLoader(Dataset):
    def __init__(self, 
                 root_path: str, 
                 class_type: str, 
                 trial_type: str,
                 connection: str,
                 track_quality=True,
                 visualize=False):        
        
        self.root_path = Path(root_path)
        self.sessions = [d for d in sorted(self.root_path.iterdir()) if d.is_dir()]
        self.track_quality = track_quality
        self.visualize = visualize
        
        if self.track_quality:
            self.quality_analyzer = EEGQualityAnalyzer(track_quality=track_quality, visualize=visualize)
        
        self.selected_electrodes = {
            'all': ["Cz", "CP1", "C4", "Pz", "O1", "Fp1", "CP6", "O2", "CP2", "P8", "C3", "P7", "Fp2", "F8", "CP5", "F7",
                    "HEOL", "FC5", "HEOR", "P4", "T7", "F3", "T8", "FC2", "PO4", "FC6", "P3", "PO3", "Oz", "F4", "FC1", "Fz",
                    "FC4", "Fpz", "P6", "FCz", "C6", "POz", "F6", "PO6", "PO8", "C2", "TP8", "F2", "FT8", "AF4", "AF8", "CP4",
                    "AF7", "CP3", "FT7", "AF3", "TP7", "F1", "PO7", "C1", "F5", "PO5", "C5", "ECG", "P5", "VEOL", "FC3","VEOU"],
            'AK-SREP': ["F8", "FCz", "Fp1", "AF7", "AF3", "C1", "FC4", "F1", "Pz", "F2", "P5", "P6"],
            'CTK-SREP': ["F1", "P5", "F4", "AF7", "PO5", "FC1", "FCz", "Fp2", "Fz", "PO3", "TP8", "F4"],
            'AK-SRES': ["AF8", "FC6", "F8", "T8", "C6", "AF7", "Fp2", "CP6", "Fpz", "F4", "TP8", "P6"],
            'CTK-SRES': ["F1", "P4", "F4", "AF3", "CP6", "P5", "Fz", "FC2", "FC1", "P6", "PC6", "F2"]
        }
        self.data, self.labels = self.get_trials(class_type=class_type, trial_type=trial_type)
        self.calculator = GMatrixCalculator(method=connection, fs=500)
        
        if self.track_quality:
            self.quality_analyzer.print_quality_summary()
        
    def load_eeg_data(self, parquet_file: str):
        return pd.read_csv(parquet_file)

    def _downsample(self, data, factor=2):
        """Downsamples EEG data by a specified factor using mean aggregation."""
        valid_length = (data.shape[2] // factor) * factor
        truncated = data[:, :, :valid_length]
        new_length = valid_length // factor
        return truncated.reshape(data.shape[0], data.shape[1], new_length, factor).mean(axis=-1)

    def _apply_butterworth_filter(self, data, low_freq=8, high_freq=30, sampling_rate=1000, order=5):
        """Apply butterworth bandpass filter to EEG data."""
        nyq = 0.5 * sampling_rate
        low = low_freq / nyq
        high = high_freq / nyq

        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data, axis=-1)

        return filtered_data

    def _min_max_normalization_per_signal(self, data, feature_range=(0, 1)):
        """Applies min-max normalization to EEG data on a per-signal basis."""      
        min_val, max_val = feature_range
        data_min = np.min(data, axis=-1, keepdims=True)
        data_max = np.max(data, axis=-1, keepdims=True)
        data_range = data_max - data_min
        
        data_range[data_range == 0] = 1
        
        normalized_data = (data - data_min) / data_range
        normalized_data = normalized_data * (max_val - min_val) + min_val
        return normalized_data

    def extract_trials(self, df, class_type="AK-SREP", trial_type="reading", target_samples=4000):
        """Extracts EEG trials and labels from a DataFrame."""
        event_column = df.iloc[:, -1] 
        
        selected_electrodes = self.selected_electrodes[class_type]

        trials = []
        labels = []

        start_event = 21 if trial_type == "reading" else 30

        event_indices = event_column[event_column.notna()].index.tolist()
        event_values = event_column.dropna().values

        for i in range(len(event_values)):
            if event_values[i] == start_event:  
                if i == 0:
                    continue  
                
                if trial_type == "reading":
                    label = event_values[i - 1]  
                else:
                    label = event_values[i - 3]
                start_idx = event_indices[i]  
                if i + 1 < len(event_indices):
                    end_idx = event_indices[i + 1]
                else:
                    continue
                
                trial_data = df.loc[start_idx:end_idx, selected_electrodes].values.T  
                current_samples = trial_data.shape[1]
                if current_samples < target_samples:    
                    padding = np.zeros((trial_data.shape[0], target_samples - current_samples))
                    trial_data = np.hstack((trial_data, padding))
                elif current_samples > target_samples:
                    trial_data = trial_data[:, :target_samples]

                trials.append(trial_data)
                labels.append(label)
        
        return np.array(trials), np.array(labels)

    def get_trials(self, class_type="AK-SREP", trial_type="reading"):
        """Extracts and processes all trials from all EEG files."""
        all_trials = []
        all_labels = []

        for session in tqdm(self.sessions, desc="Processing Sessions"):
            eeg_files = list(session.glob("*.csv"))
            for eeg_file in eeg_files:
                df = self.load_eeg_data(str(eeg_file))
                trials, labels = self.extract_trials(df, class_type=class_type, trial_type=trial_type)
                all_trials.extend(trials)
                all_labels.extend(labels)

        all_trials = np.array(all_trials)
        all_labels = np.array(all_labels)
   
        if self.track_quality:
            print("\nInitial EEG data shape:", all_trials.shape)
            self.quality_analyzer.evaluate_quality(all_trials, "Raw Signal", fs=1000)
        
        filtered_data = self._apply_butterworth_filter(all_trials, low_freq=8, high_freq=30, sampling_rate=1000, order=5)
        if self.track_quality:
            self.quality_analyzer.evaluate_quality(filtered_data, "After Bandpass Filtering", fs=1000)
        
        downsampled_data = self._downsample(filtered_data, factor=2)
        if self.track_quality:
            self.quality_analyzer.evaluate_quality(downsampled_data, "After Downsampling", fs=500)  # New Fs = 1000/2 = 500
        
        normalized_data = self._min_max_normalization_per_signal(downsampled_data)
        if self.track_quality:
            self.quality_analyzer.evaluate_quality(downsampled_data, "After Normalization", fs=500)
        
        print(f"Final dataset shape: {downsampled_data.shape}, labels shape: {all_labels.shape}")

        return normalized_data, all_labels

    def __getitem__(self, idx):
        adjacency_matrix = torch.as_tensor(self.calculator.compute_G_matrix(self.data[idx]), dtype=torch.float32)
        feature_matrix = torch.as_tensor(self.data[idx].copy(), dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return feature_matrix, adjacency_matrix, label - 1
    
    def __len__(self):
        return self.data.shape[0]
    
if __name__ == '__main__':
    dataset = EEGDataLoader(root_path = "D:\Work\MachineLearning\Projects\DRAFT\Sentences",
                            class_type = "AK-SREP",
                            trial_type = "reading",
                            connection = "pearson",
                            track_quality = False,
                            visualize = False)