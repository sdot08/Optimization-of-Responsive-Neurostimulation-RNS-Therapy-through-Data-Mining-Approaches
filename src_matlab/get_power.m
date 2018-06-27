%output a table with each filename corresponds to the power of each band in 
% each channel(28 features)
data = Catalog_222;
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
        channelPower = get_power_load(path);
        channelPowers(i,:) = channelPower;
    else
        disp('not found')
    end
end
T_222 = table(data.Timestamp_int, channelPowers);