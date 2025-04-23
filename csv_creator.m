clear; clc;

[filename, pathname] = uigetfile({'*.bdf;*.edf';'*.*'}, ...
    "Please select the BDF or EDF file that needs to be converted.", 'MultiSelect', 'on');

EEG = readbdfdata(filename, pathname);
EEG_origin = EEG.data;  % EEG signal matrix (Channels x Samples)
EEG_event = EEG.event;  % Event markers

% Handle both single and multiple file selection cases
if iscell(filename)
    filename = filename{1};  
end

% Extract filename without extension
[~, name, ~] = fileparts(filename);

% Get number of channels and timepoints
[num_channels, num_samples] = size(EEG_origin);
time_vector = (0:num_samples-1)' / EEG.srate; % Convert to column vector (num_samples, 1)

% Extract original channel names from EEG struct
if isfield(EEG, 'chanlocs') && isfield(EEG.chanlocs, 'labels')
    channel_names = {EEG.chanlocs.labels}; % Extract original channel labels
    if length(channel_names) ~= num_channels
        error('❌ Number of extracted channel names does not match EEG data dimensions!');
    end
else
    error('❌ Channel names not found in EEG struct!');
end

% Transpose EEG data to match dimensions (Samples x Channels)
EEG_origin = EEG_origin'; % Now it's (num_samples, num_channels)

% Initialize event column as a cell array
event_column = repmat({''}, num_samples, 1);

% Fill event column based on event latencies
for i = 1:length(EEG_event)
    event_index = round(EEG_event(i).latency); % Get sample index
    if event_index > 0 && event_index <= num_samples
        event_column{event_index} = EEG_event(i).type; % Store event marker as a string
    end
end

% Convert EEG data to table format (Ensuring matching dimensions)
eeg_table = array2table([time_vector, EEG_origin], ...
    'VariableNames', ['Time', channel_names]);

% Add event column to table
eeg_table.Event = event_column;

% Define output filename correctly
output_filename = fullfile(pathname, strcat(name, '_EEG_with_events.csv'));

% Save to CSV
writetable(eeg_table, output_filename);

disp(['✅ EEG data with events saved to: ', output_filename]);
