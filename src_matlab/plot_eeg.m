function plot_eeg(data)
dg_max = 1016;  %graph stuff
Nchan = size(data,2); % Number of channels
extm       = []; %graph stuff
dat_means  = zeros(1, Nchan);
ytk_lb     = cell(1, Nchan); %graph stuff channel name holder
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
end

%set(gca, 'YTick', dat_means, 'YTickLabel', ytk_lb);
%title(['Patient 231, Datatime: ' num2str(label)])
%fig_name = ['/Users/hp/Desktop/summer research/ForByron_060618/fig/artifact_sample/' num2str(label) '.png'];
%export_fig(fig_name, '-a1', '-nocrop');
end