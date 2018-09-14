%% determine if the wave of the file contains simulation, 
%% and plot the wave if required 
%% input: filename,label(the time of the file), if_plot(whether to plot the EEG wave)
%% output if the EEG wave of the file contains stimulation
%% determine the logical output by look at whether if the wave contains a flat 
%% region

function eeg_s = stitch(filename, if_plot)
fs = 250; % Sampling rate

data = readPersystDat(filename);
t = (0:size(data,1)-1)/fs;

dg_max = 1016;  %graph stuff
thres = 50; %threshold

Nchan = size(data,2); % Number of channels
extm       = []; %graph stuff
dat_means  = zeros(1, Nchan);
ytk_lb     = cell(1, Nchan); %graph stuff channel name holder
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
if find_all_flag == 1
    idx_nu = [];
    win = [-20, 2*250].';
    win_all = repmat(win, 1, length(idx_ch)) + repmat(idx_ch, 2, 1);
    idx_n = [];
    for i = 1:size(win_all,2)
        idx_n = [idx_n, win_all(1,i):win_all(2,i)];
    end
    idx_nu = unique(idx_n);
    indices = idx_nu < size(data,1) & idx_nu > 0;
    
    idx_nu = idx_nu(indices);
    eeg_s = data;
    eeg_s(idx_nu,:) = [];
else
    eeg_s = data;
end
if if_plot
    f_stim = figure;
    stitch_part = figure;
% Plot raw data and detected stimulations
    for ch = 1 : Nchan
        figure(f_stim);
        dat = data(:, ch);
        eeg = eeg_s(:,ch);
        addval = (ch-1)*(dg_max+100);
        dat = dat + addval;
        eeg = eeg + addval;
        extm = [extm, min(dat), max(dat)];
        dat_means(ch) = mean(dat);
        ytk_lb{ch} = ['Ch', num2str(ch)];
        hold on, plot(t, dat, 'k'); 
        hold on, plot(t(idx_nu), dat(idx_nu), 'r');
        
        figure(stitch_part);
        hold on,plot(eeg, 'k'); 
    end
    figure(f_stim);
    xlim([t(1) t(end)]);
    xlabel('Time (sec)');
    set(gca, 'YTick', dat_means, 'YTickLabel', ytk_lb);
    %title(['Patient 231, Datatime: ' num2str(label)])
    %fig_name = ['/Users/hp/Desktop/summer research/ForByron_060618/fig/artifact_sample/' num2str(label) '.png'];
    %export_fig(fig_name, '-a1', '-nocrop');
end

