%col: col to filter
%col_save: save the value that meet the condition to col_save
%true_str: condition
% A; table that store value == true_str
% B; table that store value != true_str
% C; table that store all value

function [A, B, C] = dummy2bool(t, col, col_save, true_str)
A = [];
B = [];
C = [];
for i = 1:height(t)
    val = t{i,col};
    if i == 1490
        disp(i)
        disp(val)
    end
    if val == true_str
        A = [A, t{i, col_save}];
        if i == 1490
            disp(i)
            disp(val)
        end
    else
        B = [B, t{i, col_save}];
    end
    C = [C, t{i, col_save}];
end
    