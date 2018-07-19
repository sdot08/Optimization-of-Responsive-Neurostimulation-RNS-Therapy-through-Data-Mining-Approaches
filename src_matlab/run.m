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
plot_timestamp_stim(222, Catalog_222)

%T_222_arr = table2array(T_222);
%T_231_arr = table2array(T_231);



%save('/Users/hp/GitHub/EEG/data/features', 'T_222_arr', 'T_231_arr', '-v7.3');
features_222 = join(T_222, numi_T_222,'Keys','Var1');
features_231 = join(T_231, numi_T_231,'Keys','Var1');
features_222 = table2array(features_222(:,[2,3,5]));
features_231 = table2array(features_231(:,[2,3,5]));
%T_222_arr_scheduled = features_222(ismember(features_222(:,1), scheduled_222_ro),:);
%T_231_arr_scheduled = features_231(ismember(features_231(:,1), scheduled_231_ro),:);
T_222_arr_scheduled = features_222(~ismember(features_222(:,1), scheduled_222_ro) & ismember(features_222(:,1), scheduled_222),:);
T_231_arr_scheduled = features_231(~ismember(features_231(:,1), scheduled_231_ro) & ismember(features_231(:,1), scheduled_231),:);
T_222_arr_scheduled = features_222(ismember(features_222(:,1), scheduled_222_ro),:);
T_231_arr_scheduled = features_231(ismember(features_231(:,1), scheduled_231_ro),:);
save('/Users/hp/GitHub/EEG/data/features_sti', 'T_222_arr_scheduled', 'T_231_arr_scheduled', '-v7.3');
save('/Users/hp/GitHub/EEG/data/features', 'T_222_arr_scheduled', 'T_231_arr_scheduled', '-v7.3');
