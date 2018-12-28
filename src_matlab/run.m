%%import data from the file NYU_ECoG_Catalog.csv to table Catalog
import_data;
pat_id_list = {231};
if_le_list = {0,0,1};

for i = 1:length(pat_id_list)
    id = pat_id_list{i};   
    if_le = if_le_list{i};
    id = 231
    if_le = 0
    prepath = strcat('/Users/hp/GitHub/EEG/datdata/',num2str(id), '/');
    % convert convert the value in column 'RawLocalTimestamp' from str to
% integer
    Catalog = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', id);
    [T_power,file_le, t_le] = stitchall(Catalog,id, prepath, if_le);
    T_numi = get_numi(Catalog,id, prepath);
    %[stimulated, scheduled] = filter_scheduled(Catalog, id, prepath);
    [sche_dates, sti_dates, all_dates] = dummy2bool(Catalog, 'ECoGtrigger', 'Timestamp_int', 'Scheduled');
    T_S = get_sleep(Catalog);
    T_l = get_longepi(Catalog, if_le);
    
    features_1 = join(T_l, join(T_S, join(T_power, T_numi,'Keys','Var1'), 'Keys','Var1'), 'Keys','Var1');
    features_2 = table2array(features_1(:,[1,2,3,5,7,9]));
    features_3 = table2array(features_1(:,[2,3,5,7,9]));
    features = conv_dat2int(features_2, features_3);
    if if_le
        dates_filter = [sche_dates, t_le.'];
    else
        dates_filter = sche_dates;
    T_arr_scheduled = features(ismember(features(:,2), dates_filter),:);
    save(strcat('/Users/hp/GitHub/EEG/data/features_90', num2str(id)), 'T_arr_scheduled', '-v7.3');
end


%for patient 229

import_data;
pat_id_list = {229,222,231};

for i = 1:length(pat_id_list)
    id = pat_id_list{i}; 
    disp(id)
    id = 231;
    prepath = strcat('/Users/hp/GitHub/EEG/datdata/',num2str(id), '/');
    % convert convert the value in column 'RawLocalTimestamp' from str to
% integer
    Catalog = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', id);    
    
    T_power = stitchall(Catalog,id, prepath, 0);
    T_numi = get_numi(Catalog,id, prepath);
    %[stimulated, scheduled] = filter_scheduled(Catalog, id, prepath);
    [sche_dates, sti_dates, all_dates] = dummy2bool(Catalog, 'ECoGtrigger', 'Timestamp_int', 'Scheduled');
    T_S = get_sleep(Catalog);
    
    features_1 = join(T_S, join(T_power, T_numi,'Keys','Var1'), 'Keys','Var1');
    features_2 = table2array(features_1(:,[1,2,3,5,7]));
    features_3 = table2array(features_1(:,[2,3,5,7]));
    features = conv_dat2int(features_2, features_3);
    T_arr_scheduled = features(ismember(features(:,2), sche_dates),:);
    save(strcat('/Users/hp/GitHub/EEG/data/features_149', num2str(id)), 'T_arr_scheduled', '-v7.3');
end











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
