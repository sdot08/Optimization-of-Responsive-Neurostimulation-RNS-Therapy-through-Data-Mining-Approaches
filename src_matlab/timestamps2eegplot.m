%% plot the eeg wave corresponds to the timestamp
%% timestamp sample format : "9/23/2015 3:59:13.0"
function timestamps2eegplot(data, timestamp1, timestamp2, sche_dates, patid)
T = data.Timestamp_int;
F = data.Filename;
prepath = '/Users/hp/GitHub/EEG/datdata/';

b_minus = datenum('2000', 'yyyy');
ts1 = datenum(timestamp1, 'yyyy-mm-dd HH:MM:SS') - b_minus;
ts2 = datenum(timestamp2, 'yyyy-mm-dd HH:MM:SS') - b_minus;

for i = 1:length(T)
	% find the row with the timestamp
    if T(i) > ts1 && T(i) < ts2 && ismember(T(i), sche_dates)
        t = datetime(T(i) + b_minus,'ConvertFrom','datenum');
        t.Format = 'MM-dd-yyyy';
        filename = F(i);
        path = strcat(prepath, filename);
        plot_eeg_f(patid,filename, datestr(t))
    end
end
    