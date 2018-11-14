% 
% prepath = '/Users/hp/GitHub/EEG/datdata/229/';
% filename = '131787687158940000.dat';
% path = strcat(prepath, filename);
% data = readPersystDat(path);
% getfirstns(data, 50, 1)
%read the eeg segment file and output only the first t seconds 
function output = getfirstns(data, idx, if_plot)
fs = 250; % Sampling rate
%get the eeg segments that are 2 seconds before the stimulation if the
%segments is less than 10 seconds. if longer than 10 seconds, get the 80%
%of it.
if idx > fs * 10
    idx_u = idx - 2 * fs;
else
    idx_u = floor(idx * 0.8);
end
output = data(1:idx_u, :);


if if_plot
    plot_eeg(output)
end
end

