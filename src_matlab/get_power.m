%output a table with each filename corresponds to the power of each band in 
% each channel(28 features)
data = Catalog;
files = data.Filename;
%initiate channelPowers
channelPowers = zeros(height(data),28, 'double');
for i = 1:length(files)
    filename = files(i);
    prepath = '/Users/hp/GitHub/EEG/datdata/';
    path = strcat(prepath, filename);
    %check if the file exists
    if exist(path, 'file') == 2
        %get channel power for that particular file
        data_power  = readPersystDat(path);
        channelPower = get_power_load(data_power);
        channelPowers(i,:) = channelPower;
    else
        disp('not found')
    end
end
T_222 = table(data.Filename, data.Timestamp_int, channelPowers);

