import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

class EEGQualityAnalyzer:
    """
    A class dedicated to analyzing and visualizing EEG signal quality.
    Provides methods for calculating quality metrics and visualizing progress.
    """
    
    def __init__(self, track_quality=True, visualize=False):
        self.track_quality = track_quality
        self.visualize = visualize
        self.quality_metrics = {}
        
    def calculate_snr(self, data, fs=1000, freq_band=(8, 30)):
        """
        Calculate Signal-to-Noise Ratio focusing on the specified frequency band
        using Welch's power spectral density estimation.
        
        Returns the average SNR across all channels and trials.
        """
        n_trials, n_channels, n_samples = data.shape
        all_snrs = []
        
        for trial_idx in range(n_trials):
            trial_snrs = []
            for ch_idx in range(n_channels):
                signal = data[trial_idx, ch_idx, :]

                freqs, psd = welch(signal, fs=fs, nperseg=min(256, n_samples), scaling='density')
                
                signal_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
                noise_mask = ~signal_mask
                
                signal_power = np.mean(psd[signal_mask]) if np.any(signal_mask) else 0
                noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 0
                
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                else:
                    snr = float('inf')
                trial_snrs.append(snr)
            all_snrs.append(np.mean(trial_snrs))
        return np.mean(all_snrs)
    
    def calculate_kurtosis(self, data):
        """
        Calculate kurtosis of the data.
        Higher values indicate more outliers/artifacts.
        """
        kurt_values = []
        n_trials, n_channels, _ = data.shape
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                signal = data[trial_idx, ch_idx, :]
                kurt = np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4)
                kurt_values.append(kurt)
        return np.mean(kurt_values)
    
    def calculate_spectral_entropy(self, data, fs=1000):
        """
        Calculate normalized spectral entropy.
        Lower values indicate more regular/predictable signals.
        """
        n_trials, n_channels, _ = data.shape
        entropies = []
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                signal = data[trial_idx, ch_idx, :]
                
                # Calculate power spectral density
                freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
                
                # Normalize PSD
                psd_norm = psd / np.sum(psd)
                
                # Remove zeros to avoid log(0)
                psd_norm = psd_norm[psd_norm > 0]
                
                # Calculate entropy
                entropy = -np.sum(psd_norm * np.log2(psd_norm))
                entropies.append(entropy)
                
        return np.mean(entropies)
    
    def calculate_hjorth_parameters(self, data):
        """
        Calculate Hjorth parameters: Activity, Mobility, and Complexity.
        Good for measuring statistical properties of EEG.
        """
        n_trials, n_channels, _ = data.shape
        mobility_values = []
        complexity_values = []
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                signal = data[trial_idx, ch_idx, :]
                
                d1 = np.diff(signal)
                d2 = np.diff(d1)
                
                d1 = np.append(d1, 0)
                d2 = np.append(d2, [0, 0])
                
                activity = np.var(signal)
                mobility = np.sqrt(np.var(d1) / activity) if activity > 0 else 0
                mobility_values.append(mobility)
                
                mobility_d1 = np.sqrt(np.var(d2) / np.var(d1)) if np.var(d1) > 0 else 0
                complexity = mobility_d1 / mobility if mobility > 0 else 0
                complexity_values.append(complexity)
                
        return np.mean(mobility_values), np.mean(complexity_values)
    
    def calculate_band_power_ratio(self, data, fs=1000):
        """
        Calculate alpha-to-theta or beta-to-theta band power ratio.
        Useful for cognitive state assessment.
        """
        n_trials, n_channels, _ = data.shape
        alpha_theta_ratios = []
        beta_theta_ratios = []
        
        theta_band = (4, 8)
        alpha_band = (8, 13)
        beta_band = (13, 30)
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                signal = data[trial_idx, ch_idx, :]
                
            
                freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
                
                theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
                alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
                beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
                
                theta_power = np.mean(psd[theta_mask]) if np.any(theta_mask) else 1e-10
                alpha_power = np.mean(psd[alpha_mask]) if np.any(alpha_mask) else 0
                beta_power = np.mean(psd[beta_mask]) if np.any(beta_mask) else 0
                
                alpha_theta_ratio = alpha_power / theta_power
                beta_theta_ratio = beta_power / theta_power
                
                alpha_theta_ratios.append(alpha_theta_ratio)
                beta_theta_ratios.append(beta_theta_ratio)
                
        return np.mean(alpha_theta_ratios), np.mean(beta_theta_ratios)
    
    def evaluate_quality(self, data, stage_name, fs=1000):
        """Evaluate signal quality using multiple metrics and store results."""
        if not self.track_quality:
            return
            
        snr = self.calculate_snr(data, fs=fs)
        kurtosis = self.calculate_kurtosis(data)
        spectral_entropy = self.calculate_spectral_entropy(data, fs=fs)
        mobility, complexity = self.calculate_hjorth_parameters(data)
        alpha_theta_ratio, beta_theta_ratio = self.calculate_band_power_ratio(data, fs=fs)
        
        self.quality_metrics[stage_name] = {
            'SNR (dB)': snr,
            'Kurtosis': kurtosis,
            'Spectral Entropy': spectral_entropy,
            'Hjorth Mobility': mobility,
            'Hjorth Complexity': complexity,
            'Alpha/Theta Ratio': alpha_theta_ratio,
            'Beta/Theta Ratio': beta_theta_ratio
        }
        
        if self.visualize:
            self.visualize_sample(data, stage_name)
            
        print(f"\n--- Quality metrics after {stage_name} ---")
        print(f"SNR: {snr:.2f} dB")
        print(f"Kurtosis: {kurtosis:.2f}")
        print(f"Spectral Entropy: {spectral_entropy:.4f}")
        print(f"Hjorth Mobility: {mobility:.4f}")
        print(f"Hjorth Complexity: {complexity:.4f}")
        print(f"Alpha/Theta Ratio: {alpha_theta_ratio:.4f}")
        print(f"Beta/Theta Ratio: {beta_theta_ratio:.4f}")
    
    def visualize_sample(self, data, stage_name, trial_idx=0, channel_idx=0):
        """Visualize a sample of data for quality inspection."""
        if not self.visualize:
            return
            
        plt.figure(figsize=(12, 6))
        signal = data[trial_idx, channel_idx, :]
        plt.plot(signal)
        plt.title(f"Sample EEG signal after {stage_name}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(f"eeg_quality_{stage_name}.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        freqs, psd = welch(signal, fs=500, nperseg=min(256, len(signal)))
        plt.semilogy(freqs, psd)
        plt.title(f"Power Spectral Density after {stage_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.tight_layout()
        plt.savefig(f"eeg_psd_{stage_name}.png")
        plt.close()
    
    def print_quality_summary(self):
        """Print a summary of quality improvements across processing stages."""
        if not self.track_quality or not self.quality_metrics:
            return
            
        print("\n=== QUALITY IMPROVEMENT SUMMARY ===")
        

        all_stages = list(self.quality_metrics.keys())
        if len(all_stages) <= 1:
            print("Not enough processing stages to compare.")
            return
            
        baseline_stage = all_stages[0]
        final_stage = all_stages[-1]
        
        print(f"\nComparison from {baseline_stage} to {final_stage}:")
        
        for metric in self.quality_metrics[baseline_stage].keys():
            baseline_value = self.quality_metrics[baseline_stage][metric]
            final_value = self.quality_metrics[final_stage][metric]
            
            if metric == 'SNR (dB)' or metric == 'RMS' or metric.endswith('Ratio'):
                change = final_value - baseline_value
                percent = (change / abs(baseline_value)) * 100 if baseline_value != 0 else float('inf')
                change_desc = "improved" if change > 0 else "worsened"
            elif metric == 'Kurtosis':
                change = baseline_value - final_value
                percent = (change / abs(baseline_value)) * 100 if baseline_value != 0 else float('inf')
                change_desc = "improved" if change > 0 else "worsened"
            else:
                change = final_value - baseline_value
                percent = (change / abs(baseline_value)) * 100 if baseline_value != 0 else float('inf')
                change_desc = "changed by"
                
            print(f"{metric}: {baseline_value:.4f} → {final_value:.4f} ({change_desc} {abs(percent):.1f}%)")
        
        if self.visualize:
            self.plot_quality_comparison()
    
    def plot_quality_comparison(self):
        """Plot metrics across processing stages for visualization."""
        if not self.visualize or not self.quality_metrics:
            return
            
        all_stages = list(self.quality_metrics.keys())
        metrics = list(self.quality_metrics[all_stages[0]].keys())
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            values = [self.quality_metrics[stage][metric] for stage in all_stages]
            plt.plot(all_stages, values, 'o-')
            plt.title(f"Progression of {metric} across processing stages")
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"quality_progression_{metric.replace('/', '_')}.png")
            plt.close()