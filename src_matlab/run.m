%%import data from the file NYU_ECoG_Catalog.csv to table Catalog
import_data;
% convert convert the value in column 'RawLocalTimestamp' from str to
% integer for patient 222 and patient 231
Catalog_222 = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', 222);
Catalog_231 = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', 231);
%summary_data(Catalog);
%get the date in which EEG curve contains stimulation
[date_222, temp] = stim_date_func(Catalog_222);
[date_231, temp] = stim_date_func(Catalog_231);
plot_timestamp_stim(231, Catalog_231)


T_222_arr = table2array(T_222);
T_231_arr = table2array(T_231);
save('/Users/hp/GitHub/EEG/data/features', 'T_222_arr', 'T_231_arr', '-v7.3');
