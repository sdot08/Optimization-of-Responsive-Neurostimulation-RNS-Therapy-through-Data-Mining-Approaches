%output a table contain the feature of whether 
function T_S = get_longepi(data)
warning('off','all')
[filter_longepi, ph, ph] = dummy2bool(data, 'ECoGtrigger', 'Filename', 'Long_Episode');
files = data.Filename;

dummies = zeros(height(data),1, 'double');
for i = 1:length(files)
    file = files(i);
    if ismember(file,filter_longepi)
        dummy = 1;
    else
        dummy = 0;
    end
    dummies(i) = dummy;
end
T_S = table(data.Filename, data.Timestamp_int, dummies);

%save('/Users/hp/GitHub/EEG/data/numi', 'numi_T_231', 'numi_T_222', '-v7.3');
