import_data;
%% convert the value in column 'RawLocalTimestamp' from str to
%% integer
Catalog = preprocess_time2int(Catalog, 'RawLocalTimestamp', 231);

for i = 1:4
    ta1 = ta(:,i);
    figure;
    plot(1:length(ta1),zscore(ta1,1))
end
