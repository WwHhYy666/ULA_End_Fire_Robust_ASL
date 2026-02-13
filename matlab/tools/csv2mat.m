clear; clc; close all;

%% 1. Configuration
% Define the input CSV path
csvPath = 'D:\acoustic\asl\EXPERIMENT\End-Fire\DOA_results_per_file.csv';
outputDir = fileparts(csvPath); % Save .mat files in the same folder as CSV

%% 2. Read the Data
if ~isfile(csvPath)
    error('File not found: %s', csvPath);
end

fprintf('Reading CSV file...\n');
% Read table, ensuring Filename is treated as text (string)
opts = detectImportOptions(csvPath);
opts.VariableTypes{1} = 'string'; % Force first column (Filename) to be string
FullTable = readtable(csvPath, opts);

fprintf('Total rows loaded: %d\n', height(FullTable));

%% 3. Classify Data (1m vs 2m)
% The convention is "20d1m_..." or "100d2m_..."
% We look for the substring "d1m" or "d2m" in the Filename column.

% Filter for 1m
idx_1m = contains(FullTable.Filename, 'd1m', 'IgnoreCase', true);
Table_1m = FullTable(idx_1m, :);

% Filter for 2m
idx_2m = contains(FullTable.Filename, 'd2m', 'IgnoreCase', true);
Table_2m = FullTable(idx_2m, :);

%% 4. Save to .mat files
% Define output filenames
outFile_1m = fullfile(outputDir, 'DOA_results_1m.mat');
outFile_2m = fullfile(outputDir, 'DOA_results_2m.mat');

% Save 1m Data
if ~isempty(Table_1m)
    save(outFile_1m, 'Table_1m');
    fprintf('Saved 1m data (%d rows) to: %s\n', height(Table_1m), outFile_1m);
else
    warning('No 1m data found (filenames containing "d1m").');
end

% Save 2m Data
if ~isempty(Table_2m)
    save(outFile_2m, 'Table_2m');
    fprintf('Saved 2m data (%d rows) to: %s\n', height(Table_2m), outFile_2m);
else
    warning('No 2m data found (filenames containing "d2m").');
end

fprintf('Done.\n');