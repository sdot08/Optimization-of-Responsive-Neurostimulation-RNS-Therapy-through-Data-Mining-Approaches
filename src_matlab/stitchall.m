%output a table with each filename corresponds to the power of each band in 
% each channel(28 features)
function T = stitchall(data)
files = data.Filename;
channelPowers = zeros(height(data),28, 'double');
numi = zeros(height(data),2, 'double');
for i = 1:length(files)
    filename = files(i);
    prepath = '/Users/hp/GitHub/EEG/datdata/';
    path = strcat(prepath, filename);
    %check if the file exists
    %path = '/Users/hp/GitHub/EEG/datdata/130901185189430000.dat';

    if exist(path, 'file') == 2
        %get channel power for that particular file
        data_eeg = stitch(path, 0);
        [numi(i,1), numi(i,2)] = interictalplot(data_eeg, 0);
        channelPower = get_power_load(data_eeg);
        channelPowers(i,:) = channelPower;
    else
        disp('not found')
    end
end
T = table(data.Filename, data.Timestamp_int, channelPowers, numi);

