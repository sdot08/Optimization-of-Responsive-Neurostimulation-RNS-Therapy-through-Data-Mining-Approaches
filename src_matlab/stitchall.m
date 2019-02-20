%output a table with each filename corresponds to the power of each band in 
% each channel(28 features)
function [T,file_le, t_le] = stitchall(data, id, prepath, if_le, high_cut)
% data = Catalog;
% id = 231;
% if_le = 0;
% high_cut = 90;

files = data.Filename;

channelPowers = zeros(height(data),28, 'double');

[idxs, file_le, t_le] = stim_firstns(data, prepath);


for i = 1:length(files)
    filename = files(i);
    path = strcat(prepath, filename);
    %check if the file exists
    %path = '/Users/hp/GitHub/EEG/datdata/231/130901185189430000.dat';
    if exist(path, 'file') == 2
        data_eeg = readPersystDat(path);
        %get channel power for that particular file
        if if_le
            if ismember(filename, file_le)
                idxx = find(file_le == filename);
                idx = idxs(idxx);
                data_eeg = getfirstns(data_eeg, idx, 0);
            end
        end
        data_eeg2 = stitch(data_eeg, 0, id);
        channelPower = get_power_load(data_eeg2, if_le, high_cut);
        channelPowers(i,:) = channelPower;
    else
        disp(filename)
        disp(i)
    end
end
T = table(data.Filename, data.Timestamp_int, channelPowers);

