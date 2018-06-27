
%% input table, convert the value in column 'RawLocalTimestamp' from str to
%% integer, the integer value will be in the new column 'Timestamp_int'

function t2 = preprocess_time2int(t1, col, patientid)
a = table2array(t1(:,col));
b = [[]];
%set pivotal year
b_minus = datenum('2000', 'yyyy');
for i = 1:length(a)
   ai = a(i);
   b(i,1) = datenum(ai, 'mm/dd/yyyy HH:MM:SS') - b_minus;
end

t1.Timestamp_int = b;
t2 = t1(t1.Initials == patientid, :);