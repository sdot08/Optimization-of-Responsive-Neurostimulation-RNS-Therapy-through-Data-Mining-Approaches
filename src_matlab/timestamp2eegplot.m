%% plot the eeg wave corresponds to the timestamp
%% timestamp sample format : "9/23/2015 3:59:13.0"
function timestamp2eegplot(data, timestamp)
T = data.Timestamp_int;
F = data.Filename;
prepath = '../datdata/';
    
for i = 1:length(T)
	% find the row with the timestamp
    if T(i) == timestamp
        filename = F(i);
        path = strcat(prepath, filename);
        stim_detection_c(path, 1, '')
    end
end
    