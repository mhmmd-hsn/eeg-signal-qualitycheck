import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew
from pywt import wavedec
from tqdm import tqdm
import os

class EEGFeatureExtractor:
    def __init__(self, fs=500, freq_bands=None):
        self.fs = fs
        if freq_bands is None:
            self.freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30)
            }
        else:
            self.freq_bands = freq_bands
    
    def extract_features(self, eeg_signal):
        num_samples, num_channels, num_timepoints = eeg_signal.shape
        
        all_features = []
        feature_names = []
        
        for sample_idx in tqdm(range(num_samples), desc="Extracting features"):
            sample_features = []
            sample_data = eeg_signal[sample_idx]
            
            for channel_idx in range(num_channels):
                channel_data = sample_data[channel_idx]
                
                time_features = self._extract_time_features(channel_data)
                spectral_features = self._extract_spectral_features(channel_data)
                
                channel_features = np.concatenate([time_features, spectral_features])
                sample_features.append(channel_features)
                
                if sample_idx == 0:  
                    time_feature_names = self._get_time_feature_names(channel_idx)
                    spectral_feature_names = self._get_spectral_feature_names(channel_idx)
                    feature_names.extend(time_feature_names + spectral_feature_names)
            
            if num_channels > 1:
                connectivity_features = self._extract_connectivity_features(sample_data)
                
                if sample_idx == 0:
                    connectivity_feature_names = self._get_connectivity_feature_names(num_channels)
                    feature_names.extend(connectivity_feature_names)
                
                sample_features = np.concatenate(sample_features)
                features_vector = np.concatenate([sample_features, connectivity_features])
            else:
                features_vector = np.concatenate(sample_features)
            
            all_features.append(features_vector)
        
        return np.array(all_features), feature_names
    
    def _extract_time_features(self, channel_data):
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        skewness = skew(channel_data)
        peak_to_peak = np.max(channel_data) - np.min(channel_data)
        zero_crossings = np.sum(np.diff(np.signbit(channel_data)))
        
        return np.array([mean, std, skewness, peak_to_peak, zero_crossings])
    
    def _get_time_feature_names(self, channel_idx):
        return [
            f"ch{channel_idx}_mean",
            f"ch{channel_idx}_std",
            f"ch{channel_idx}_skewness",
            f"ch{channel_idx}_peak_to_peak",
            f"ch{channel_idx}_zero_crossings"
        ]
    
    def _extract_spectral_features(self, channel_data):
        freqs, psd = signal.welch(channel_data, fs=self.fs, nperseg=min(256, len(channel_data)))
        
        features = []
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            low_idx = np.argmax(freqs >= low_freq) if any(freqs >= low_freq) else len(freqs)-1
            high_idx = np.argmax(freqs >= high_freq) if any(freqs >= high_freq) else len(freqs)-1
            
            band_power = np.sum(psd[low_idx:high_idx+1])
            
            features.append(band_power)
        
        total_power = np.sum(psd)
        relative_band_powers = []
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            low_idx = np.argmax(freqs >= low_freq) if any(freqs >= low_freq) else len(freqs)-1
            high_idx = np.argmax(freqs >= high_freq) if any(freqs >= high_freq) else len(freqs)-1
            
            band_power = np.sum(psd[low_idx:high_idx+1])
            relative_power = band_power / total_power if total_power > 0 else 0
            relative_band_powers.append(relative_power)
        
        spectral_entropy = self._spectral_entropy(psd)
        
        features.extend(relative_band_powers)
        features.append(spectral_entropy)
        
        return np.array(features)
    
    def _get_spectral_feature_names(self, channel_idx):
        feature_names = []
        
        for band_name in self.freq_bands.keys():
            feature_names.append(f"ch{channel_idx}_{band_name}_abs_power")
        
        for band_name in self.freq_bands.keys():
            feature_names.append(f"ch{channel_idx}_{band_name}_rel_power")
        
        feature_names.append(f"ch{channel_idx}_spectral_entropy")
        
        return feature_names
    
    def _extract_connectivity_features(self, sample_data):
        num_channels = sample_data.shape[0]
        
        if num_channels <= 1:
            return np.array([])
        
        features = []
        
        corr_matrix = np.corrcoef(sample_data)
        corr_features = corr_matrix[np.triu_indices(num_channels, k=1)]
        features.extend(corr_features)
        
        return np.array(features)
    
    def _get_connectivity_feature_names(self, num_channels):
        feature_names = []
        
        for i in range(num_channels):
            for j in range(i+1, num_channels):
                feature_names.append(f"corr_ch{i}_ch{j}")
        
        return feature_names
    
    def _spectral_entropy(self, psd):
        norm_psd = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        entropy = -np.sum(norm_psd * np.log2(norm_psd + 1e-10))
        return entropy
    
    def extract_and_save_to_csv(self, eeg_signal, output_file_path):
        features, feature_names = self.extract_features(eeg_signal)
        
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df = pd.DataFrame(features, columns=feature_names)
        df.to_csv(output_file_path, index=False, float_format='%.4f')
        
        return df