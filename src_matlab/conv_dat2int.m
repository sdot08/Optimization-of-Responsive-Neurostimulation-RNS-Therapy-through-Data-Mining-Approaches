%convert filename to number 
function dat2 = conv_dat2int(dat1, dat2)
    A = dat1(:,1);
    A = arrayfun(@(x) char(x), A,'UniformOutput',0);
    %A = arrayfun(@(x) char(x), A,'UniformOutput',0);
    A = cellfun(@(x) x(1:end-4), A, 'UniformOutput',0);
    B = cellfun(@(x) str2num(x), A, 'UniformOutput',0);
    B = cell2mat(B);
    dat2 = [B,dat2];