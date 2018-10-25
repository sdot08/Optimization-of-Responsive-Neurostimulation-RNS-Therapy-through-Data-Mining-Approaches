import_data;
id = 229;    
Catalog = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', id);
prepath = strcat('/Users/hp/GitHub/EEG/datdata/',num2str(id), '/');
filename = '131787687158940000.dat';
path = strcat(prepath, filename);
data = readPersystDat(path);
[idxs, file_le] = stim_firstns(Catalog, prepath);
data = readPersystDat(path);
if ismember(filename, file_le)
    data = getfirstns(data, 800,1);
end