function [A, B, C] = dummy2bool(t, col, col_save, true_str)
A = [];
B = [];
C = [];
for i = 1:height(t)
    val = t{i,col};
    if val == true_str
        A = [A, t{i, col_save}];
    else
        B = [B, t{i, col_save}];
    end
    C = [C, t{i, col_save}];
end
    