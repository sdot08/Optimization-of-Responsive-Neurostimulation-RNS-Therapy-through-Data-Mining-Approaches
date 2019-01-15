function T_S = get_sleep(data, pat)
warning('off','all')
t = data.Timestamp_int;
dates = floor(t);
times = t - dates;
thres_sleep = [0,8/24];
if pat == 241
    thres_sleep = [20/24,8/24];
    dummies = zeros(height(data),1, 'double');
    for i = 1:length(times)
        time = times(i);
        if time > thres_sleep(1) || time < thres_sleep(2)
            dummy = 1;
        else
            dummy = 0;
        end
        dummies(i) = dummy;
    end
else
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
end
T_S = table(data.Filename, data.Timestamp_int, dummies);

%save('/Users/hp/GitHub/EEG/data/numi', 'numi_T_231', 'numi_T_222', '-v7.3');
