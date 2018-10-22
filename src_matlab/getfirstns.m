% 
% prepath = '/Users/hp/GitHub/EEG/datdata/229/';
% filename = '131787687158940000.dat';
% path = strcat(prepath, filename);
% data = readPersystDat(path);
% getfirstns(data, 50, 1)
%read the eeg segment file and output only the first t seconds 
function output = getfirstns(data, t_c, if_plot)

fs = 250; % Sampling rate
idx = t_c * fs;

output = data(1:idx, :);


if if_plot
    plot_eeg(output)
end
end

