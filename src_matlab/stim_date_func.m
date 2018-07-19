%% produce all timestamps of stimulated instances and scheduled instances 
%% input Catalog_222 or Catalog_231
function [stimulated, scheduled] = stim_date_func(data_raw)
data = data_raw(:,[6, 7, end] );

prepath = '/Users/hp/GitHub/EEG/datdata/';
b = zeros(height(data), 1);
for i = 1:height(data)
    
    datai =data(i,:);
    if datai.ECoGtype == 'real-time'
        b(i) = 1;
        continue
    end
    
    
    filename = datai.Filename;
    path = strcat(prepath, filename);
    z = stim_detection_z(path, 0, '');
    %if z == true
   %     b(i) = 1;
    %    continue
  %  end
    %determine if the file correponds to path is stimulated instances or not
    b(i) = stim_detection_c(path, 0, '');
   
end
b = b == 1;
%stimulated = data_raw(b,:).RawLocalTimestamp;
stimulated = data_raw(b,:).Timestamp_int;
nb = not (b);
scheduled = data_raw(nb,:).Timestamp_int;