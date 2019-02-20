%remembember to change the _g
id = 231
import_data;
prepath = strcat('/Users/hp/GitHub/EEG/datdata/',num2str(id), '/');
Catalog = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', id);
[sche_dates, sti_dates, all_dates] = dummy2bool(Catalog, 'ECoGtrigger', 'Timestamp_int', 'Scheduled');

%timestamps2eegplot(Catalog,'2017-09-12 00:00:00', '2017-10-13 00:00:00', sche_dates, 231)
timestamps2eegplot(Catalog,'2017-11-13 00:00:00', '2017-12-14 00:00:00', sche_dates, 231)


id = 241
import_data;
prepath = strcat('/Users/hp/GitHub/EEG/datdata/',num2str(id), '/');
Catalog = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', id);
[sche_dates, sti_dates, all_dates] = dummy2bool(Catalog, 'ECoGtrigger', 'Timestamp_int', 'Scheduled');
%timestamps2eegplot(Catalog,'2018-05-07 00:00:00', '2018-06-05 00:00:00', sche_dates, 241)
timestamps2eegplot(Catalog,'2018-06-05 00:00:00', '2018-07-04 00:00:00', sche_dates, 241)

