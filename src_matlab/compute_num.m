%compute number of data points for each patient

 [T_power,file_le, t_le] = stitchall(Catalog,id, prepath, if_le, highcut);

import_data;

id = 231;
prepath = strcat('/Users/hp/GitHub/EEG/datdata/',num2str(id), '/');
if_le = 0;
high_cut = 90;
% timestamp1 = '2017-02-07 00:00:00'; 
% timestamp2 = '2018-02-21 00:00:00';
timestamp1 = '2017-11-14 00:00:00'; 
timestamp2 = '2018-10-04 00:00:00';
b_minus = datenum('2000', 'yyyy');
ts1 = datenum(timestamp1, 'yyyy-mm-dd HH:MM:SS') - b_minus;
ts2 = datenum(timestamp2, 'yyyy-mm-dd HH:MM:SS') - b_minus;

Catalog = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', id);
[sche_dates, sti_dates, all_dates] = dummy2bool(Catalog, 'ECoGtrigger', 'Timestamp_int', 'Scheduled');

data = Catalog;
files = data.Filename;
T = data.Timestamp_int;


channelPowers = zeros(height(data),28, 'double');

[idxs, file_le, t_le] = stim_firstns(data, prepath);

eeg_length = [];
for i = 1:length(files)
    filename = files(i);
    path = strcat(prepath, filename);
    %check if the file exists
    %path = '/Users/hp/GitHub/EEG/datdata/231/130901185189430000.dat';
    if exist(path, 'file') == 2
        data_eeg = readPersystDat(path);
        %get channel power for that particular file
        if if_le
            if ismember(filename, file_le)
                idxx = find(file_le == filename);
                idx = idxs(idxx);
                data_eeg = getfirstns(data_eeg, idx, 0);
            end
        end
        data_eeg2 = stitch(data_eeg, 0, id);
        

        if T(i) > ts1 && T(i) < ts2 && ismember(T(i), sche_dates)
            eeg_length = [eeg_length, length(data_eeg2)];
        end
%         channelPower = get_power_load(data_eeg2, if_le, high_cut);
%         channelPowers(i,:) = channelPower;
    else
        disp(filename)
        disp(i)
    end
end
% T = table(data.Filename, data.Timestamp_int, channelPowers);

