%output a table with each filename corresponds to the power of each band in 
% each channel(28 features)
function T = stitchall(data, id, prepath, if_le)
%data = Catalog_222;
files = data.Filename;
channelPowers = zeros(height(data),28, 'double');
if if_le
    temp, file_le = stim_firstns(Catalog, prepath);
end
for i = 1:length(files)
    filename = files(i);
    path = strcat(prepath, filename);
    %check if the file exists
    %path = '/Users/hp/GitHub/EEG/datdata/130901185189430000.dat';
    if exist(path, 'file') == 2
        data = readPersystDat(path);
        %get channel power for that particular file
        if if_le
            if ismember(filename, file_le)
                data = getfirstns(data);
            end
        end
        data_eeg = stitch(data, 0, id);
        channelPower = get_power_load(data_eeg);
        channelPowers(i,:) = channelPower;
    else
        disp(filename)
        disp(i)
    end
end
T = table(data.Filename, data.Timestamp_int, channelPowers);

