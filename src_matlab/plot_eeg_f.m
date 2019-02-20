function plot_eeg_f(id, filename, label)

%plot_eeg_f(231, '131519877860300000.dat', 1)
% path = '/Users/hp/GitHub/EEG/datdata/231/131519877860300000.dat';
% data = readPersystDat(path);
prepath = strcat('/Users/hp/GitHub/EEG/datdata/',num2str(id), '/');
path = strcat(prepath, filename);
data = readPersystDat(path);
fs = 250; % Sampling rate

t = (0:size(data,1)-1)/fs;

dg_max = 1016;  %graph stuff

Nchan = size(data,2); % Number of channels

% calculate the first order difference of the wave for each channel
data_diff = diff(data,[],1); 
% produce a matrix in which each row indicated if 
% all four channels in data_diff is all 0 or not. 1 in diff_logical indicate 
% they are all 0, and 0 indicate otherwise.
diff_logical = all(data_diff == 0, 2);
% check if the number of consecutive 0s in diff_logical is more than threshold
% if so, output true, else, output false 
idx_ch = [];
find_flag = 0;
find_all_flag = 0;
thres = 50;
for i = 1:length(diff_logical)-thres
    if find_flag == 1 && diff_logical(i) == 1
        continue
    else
        find_flag = 0;
    end
    if all(diff_logical(i:i+thres) == 1) 
        idx_ch = [idx_ch, i];
        find_flag = 1;
        find_all_flag = 1;
    end
end
idx_nu = [];
if find_all_flag == 1
    idx_nu = [];
    win = [-20, 250 * 2].';
    win_all = repmat(win, 1, length(idx_ch)) + repmat(idx_ch, 2, 1);
    idx_n = [];
    for i = 1:size(win_all,2)
        idx_n = [idx_n, win_all(1,i):win_all(2,i)];
    end
    idx_nu = unique(idx_n);
    indices = idx_nu < size(data,1) & idx_nu > 0;
    
    idx_nu = idx_nu(indices);

end

dg_max = 1016;  %graph stuff
Nchan = size(data,2); % Number of channels
extm       = []; %graph stuff
dat_means  = zeros(1, Nchan);
ytk_lb     = cell(1, Nchan); %graph stuff channel name holder
t = (0:size(data,1)-1);
f_stim = figure;

for ch = 1 : Nchan
    figure(f_stim);
    xlabel('Time (sec)/250');
    dat = data(:, ch);
    eeg = data(:,ch);
    addval = (ch-1)*(dg_max+100);
    dat = dat + addval;
    eeg = eeg + addval;
    extm = [extm, min(dat), max(dat)];
    dat_means(ch) = mean(dat);
    ytk_lb{ch} = ['Ch', num2str(ch)];
    hold on,plot(eeg, 'k'); 
    sz = 2;
    c = 'r';
    hold on, scatter(t(idx_nu), eeg(idx_nu),sz,c,'filled');
end

%set(gca, 'YTick', dat_means, 'YTickLabel', ytk_lb);
%title(['Patient 231, Datatime: ' num2str(label)])
    fig_name = strcat('/Users/hp/GitHub/EEG/fig/EEG segments/', num2str(id), '_g', '/', label, '.png');
    %fig_name = strcat('/Users/hp/GitHub/EEG/fig/EEG segments/', num2str(id), '/1.png');
    %disp(fig_name)
    export_fig(fig_name, '-a1', '-nocrop');

end