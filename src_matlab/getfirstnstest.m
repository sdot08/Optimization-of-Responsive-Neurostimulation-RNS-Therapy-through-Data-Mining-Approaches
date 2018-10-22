filename = '131787687158940000.dat';
path = strcat(prepath, filename);
data = readPersystDat(path);
[temp, file_le, idxs] = stim_firstns(Catalog, prepath);
data = readPersystDat(path);
if ismember(filename, file_le)
    i = find()
    data = getfirstns(data, );
end