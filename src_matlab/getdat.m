%% produce output features for machine learning, feature including power in
%% 7 different power band in each four channels.
files = Catalog.Filename;
channelPowers = zeros(height(Catalog),28, 'double');
for i = 1:length(files)
    %disp(i)
    filename = files(i);
    prepath = '../datdata/';
    path = strcat(prepath, filename);
    
    %disp(path)
    if exist(path, 'file') == 2
        %get the power feature for the specific file
        channelPower = get_power(path);
        channelPowers(i,:) = channelPower;
    else
        disp('not found')
    end
end
%output feature table
T = table(Catalog.Filename, channelPowers);