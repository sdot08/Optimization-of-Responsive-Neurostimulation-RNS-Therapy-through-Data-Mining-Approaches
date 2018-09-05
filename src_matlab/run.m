%%import data from the file NYU_ECoG_Catalog.csv to table Catalog
import_data;
% convert convert the value in column 'RawLocalTimestamp' from str to
% integer for patient 222 and patient 231
Catalog_222 = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', 222);
Catalog_231 = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', 231);
stitchall %for both 231 and 222
get_numi %for both 231 and 222

[stimulated_231, scheduled_231] = filter_scheduled(Catalog_231, 231);
[stimulated_222, scheduled_222] = filter_scheduled(Catalog_222, 222);

features_222_1 = join(T_222, numi_T_222,'Keys','Var1');
features_231_1 = join(T_231, numi_T_231,'Keys','Var1');
features_222_2 = table2array(features_222_1(:,[1,2,3,5]));
features_231_2 = table2array(features_231_1(:,[1,2,3,5]));
features_222_3 = table2array(features_222_1(:,[2,3,5]));
features_231_3 = table2array(features_231_1(:,[2,3,5]));
features_222 = conv_dat2int(features_222_2, features_222_3);
features_231 = conv_dat2int(features_231_2, features_231_3);


T_222_arr_scheduled_sti = features_222(~ismember(features_222(:,2), scheduled_222_ro) & ismember(features_222(:,2), scheduled_222),:);
T_231_arr_scheduled_sti = features_231(~ismember(features_231(:,2), scheduled_231_ro) & ismember(features_231(:,2), scheduled_231),:);
T_222_arr_scheduled = features_222(ismember(features_222(:,2), scheduled_222_ro),:);
T_231_arr_scheduled = features_231(ismember(features_231(:,2), scheduled_231_ro),:);
save('/Users/hp/GitHub/EEG/data/features_sti', 'T_222_arr_scheduled_sti', 'T_231_arr_scheduled_sti', '-v7.3');
save('/Users/hp/GitHub/EEG/data/features', 'T_222_arr_scheduled', 'T_231_arr_scheduled', '-v7.3');
