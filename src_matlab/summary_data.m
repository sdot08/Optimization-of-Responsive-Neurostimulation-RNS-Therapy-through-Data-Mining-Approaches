%% display a summery of the data, including head, the number of obs
%% and the number of columns
function summary_data(data)
head(data,2)
%summary(Catalog)
fprintf('the number of observation in Catalog is %i\n', height(data))
fprintf('the number of columns in Catalog is %i\n', width(data))