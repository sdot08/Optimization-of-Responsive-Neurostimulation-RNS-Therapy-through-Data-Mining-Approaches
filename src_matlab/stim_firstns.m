% has a lot of stimulation, but not in the first few seconds
% prepath = '/Users/hp/GitHub/EEG/datdata/229/';
% filename = '131787687158940000.dat';
% iend = 0;
% path = strcat(prepath, filename);
% 
% data = readPersystDat(path);
% size(data,1);
% stim_detection_c(path, 1, '', iend)

function le_
import_data;
pat_id_list = {229};

fs = 250;
le_cut = 4;
for i = 1:length(pat_id_list)
    idxs = [];
    id = pat_id_list{i};    
    prepath = strcat('/Users/hp/GitHub/EEG/datdata/',num2str(id), '/');
    Catalog = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', id);
    iend = 0;
    
    [filter_longepi, ph, ph] = dummy2bool(Catalog, 'ECoGtrigger', 'Filename', 'Long_Episode');
    Catalog_l = Catalog(ismember(Catalog(:,6),array2table(filter_longepi', 'VariableNames',{'Filename'})),:);
    files = Catalog_l.Filename;
    for i = 1:length(files)
        filename = files(i);
        path = strcat(prepath, filename);
        %check if the file exists
        %path = '/Users/hp/GitHub/EEG/datdata/130901185189430000.dat';

        if exist(path, 'file') == 2
            idx = stim_detection_c(path, 0, '', iend);
            disp(idx)
            idxs = [idxs,idx];
        else
            %disp(filename)
            disp(i)
        end
    end
    idxs = (idxs/fs)';
    fn_le = array2table(files(idxs - 2 > le_cut));
    
end

    