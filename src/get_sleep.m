%output a table with each filename corresponds to the power of each band in 
% each channel(28 features)
function T_S = get_sleep(data)
warning('off','all')
t = data.Timestamp_int;
dates = floor(t);
times = t - dates;
thres_sleep = [0,8/24];
dummies = zeros(height(data),1, 'double');
for i = 1:length(times)
    time = times(i);
    if time > thres_sleep(1) && time < thres_sleep(2)
        dummy = 1;
    else
        dummy = 0;
    end
    dummies(i) = dummy;
end
T_S = table(data.Filename, data.Timestamp_int, dummies);

%save('/Users/hp/GitHub/EEG/data/numi', 'numi_T_231', 'numi_T_222', '-v7.3');
