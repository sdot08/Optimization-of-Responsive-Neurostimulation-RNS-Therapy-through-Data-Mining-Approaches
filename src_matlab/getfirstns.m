%read the eeg segment file and output only the first t seconds 
function output = getfirstns(filename, t, if_plot)

fs = 250; % Sampling rate
idx = t * fs;
data = readPersystDat(filename);
output = data(1:idx, :);

%graph
Nchan = size(data,2); % Number of channels
extm       = []; %graph stuff
dat_means  = zeros(1, Nchan);
ytk_lb     = cell(1, Nchan); %graph stuff channel name holder
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
    title(['Patient 231, Datatime: ' num2str(label)])
    fig_name = ['/Users/hp/Desktop/summer research/ForByron_060618/fig/artifact_sample/' num2str(label) '.png'];
    %export_fig(fig_name, '-a1', '-nocrop');
end

]