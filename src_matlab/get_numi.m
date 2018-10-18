%output a table with each filename corresponds to the power of each band in 
% each channel(28 features)
function numi_T = get_numi(data, id, prepath)
warning('off','all')
%data = Catalog_222;
files = data.Filename;
%initiate channelPowers
numi = zeros(height(data),2, 'double');
for i = 1:length(files)
    filename = files(i);
    path = strcat(prepath, filename);
    %check if the file exists
    if exist(path, 'file') == 2
        %get channel power for that particular file
        data_eeg = stitch(path, 0, id);
        [numi(i,1), numi(i,2)] = interictalplot(data_eeg, 0);
    else
        disp('not found')
    end
end
numi_T = table(data.Filename, data.Timestamp_int, numi);

%save('/Users/hp/GitHub/EEG/data/numi', 'numi_T_231', 'numi_T_222', '-v7.3');
