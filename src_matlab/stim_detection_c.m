%% determine if the wave of the file contains simulation, 
%% and plot the wave if required 
%% input: filename,label(the time of the file), if_plot(whether to plot the EEG wave)
%% output if the EEG wave of the file contains stimulation
%% determine the logical output by look at whether if the wave contains a flat 
%% region

function ind = stim_detection_c(filename, if_plot, label, ns)

disp(filename)
fs = 250; % Sampling rate

data = readPersystDat(filename);
t = (0:size(data,1)-1)/fs;

dg_max = 1016;  %graph stuff
thres = 50; %threshold
if if_plot 
f_stim = figure;
end
Nchan = size(data,2); % Number of channels
extm       = []; %graph stuff
dat_means  = zeros(1, Nchan);
ytk_lb     = cell(1, Nchan); %graph stuff channel name holder
flag_stim = false; %initiate logical output to false
% calculate the first order difference of the wave for each channel
data_diff = diff(data,[],1); 
% produce a matrix in which each row indicated if 
% all four channels in data_diff is all 0 or not. 1 in diff_logical indicate 
% they are all 0, and 0 indicate otherwise.
diff_logical = all(data_diff == 0, 2);
% check if the number of consecutive 0s in diff_logical is more than threshold
% if so, output true, else, output false 
if ns == 0
    iend = length(diff_logical)-thres;
else
    iend = ns * fs;
end
ind = 0;
for i = 1:iend
    if all(diff_logical(i:i+thres) == 1) 
        flag_stim = true;
        ind = i;
        idx_ch = i:i+thres;
        break
    end
end

if if_plot
% Plot raw data and detected stimulations
    for ch = 1 : Nchan
        figure(f_stim);
        dat = data(:, ch);
        addval = (ch-1)*(dg_max+100);
        dat = dat + addval;
        extm = [extm, min(dat), max(dat)];
        dat_means(ch) = mean(dat);
        ytk_lb{ch} = ['Ch', num2str(ch)];
        hold on, plot(t, dat, 'k'); 
        if flag_stim
            hold on, plot(t(idx_ch), dat(idx_ch), 'r','Linewidth',3);
        end
    end
    figure(f_stim);
    xlim([t(1) t(end)]);
    xlabel('Time (sec)');
    set(gca, 'YTick', dat_means, 'YTickLabel', ytk_lb);
    title(['Patient 231, Datatime: ' num2str(label)])
    fig_name = ['/Users/hp/Github/EEG/fig/high_gamma_sample/' num2str(label) '.png'];
    export_fig(fig_name, '-a1', '-nocrop');
end

