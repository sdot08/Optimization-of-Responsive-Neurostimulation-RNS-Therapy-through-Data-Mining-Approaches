% has a lot of stimulation, but not in the first few seconds
% prepath = '/Users/hp/GitHub/EEG/datdata/229/';
% filename = '131787687158940000.dat';
% iend = 0;
% path = strcat(prepath, filename);
% 
% data = readPersystDat(path);
% size(data,1);
% stim_detection_c(path, 1, '', iend)

function [idxs_out, f_le, t_le] = stim_firstns(data, prepath)

fs = 250;
le_cut = 4;
iend = 0;
    
[filter_longepi, ph, ph] = dummy2bool(data, 'ECoGtrigger', 'Filename', 'Long_Episode');
data_l = data(ismember(data(:,6),array2table(filter_longepi', 'VariableNames',{'Filename'})),:);
files = data_l.Filename;
times = data_l.Timestamp_int;
idxs = [];
for i = 1:length(files)
    filename = files(i);
    path = strcat(prepath, filename);
    if exist(path, 'file') == 2
        idx = stim_detection_c(path, 0, '', iend);
        %disp(idx)
        idxs = [idxs,idx];
    else
        %disp(filename)
        %disp(i)
    end
end
idxs_m = (idxs/fs)';
t_le = times(idxs_m - 2 > le_cut);
f_le = files(idxs_m - 2 > le_cut);
idxs_out = idxs(idxs_m - 2 > le_cut);
% idxs_out = files(idxs_m - 2 > le_cut);
end

    