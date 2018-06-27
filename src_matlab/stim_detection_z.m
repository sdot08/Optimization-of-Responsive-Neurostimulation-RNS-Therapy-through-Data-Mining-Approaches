%% determine if the wave of the file contains simulation, 
%%and plot the wave if required 
%% input: filename,label(the time of the file)
%% output if the EEG wave of the file contains stimulation
%% use z score to determine the output. If there exist point whose z-score
%% exceed threshold, then output true, otherwise output false.

function flag_stim = stim_detection_z(filename, if_plot, label)

stim_thresh = 6;  %threshold in STD

fs = 250; % Sampling rate

data = readPersystDat(filename);
t = (0:size(data,1)-1)/fs; %compute the time of each point in EEG wave

dg_max = 1016;  %graph stuff

if if_plot
    f_stim = figure;
end
Nchan = size(data,2); % Number of channels
extm       = []; %graph stuff
dat_means  = zeros(1, Nchan);
ytk_lb     = cell(1, Nchan); %graph stuff channel name holder
idx_ch_all = cell(1, Nchan); %index of the start of the stimulation for 4 channels
flag_stim = false;  %initiate logical output to false
for ch = 1:Nchan
    
    % Stimulation detection
    zsr = abs(zscore(data(:,ch)));
    idx_ch = find(zsr > stim_thresh);
    idx_ch(find(diff(idx_ch)<10)+1) = [];
    idx_ch_all{ch} = idx_ch;
    
    if not (isempty(idx_ch))
       flag_stim = true;
    end
    if if_plot
    % Plot raw data and detected stimulations
        figure(f_stim);
        dat = data(:, ch);
        addval = (ch-1)*(dg_max+100);
        dat = dat + addval;
        extm = [extm, min(dat), max(dat)];
        dat_means(ch) = mean(dat);
        ytk_lb{ch} = ['Ch', num2str(ch)];
        hold on, plot(t, dat, 'k'); 
        hold on, plot(t(idx_ch), dat(idx_ch), 'ro','MarkerSize',6);
    end
end
if if_plot
    figure(f_stim);
    xlim([t(1) t(end)]);
    xlabel('Time (sec)');
    set(gca, 'YTick', dat_means, 'YTickLabel', ytk_lb);
    title(['Patient 231, Datatime: ' num2str(label)])
    fig_name = ['/Users/hp/Desktop/summer research/ForByron_060618/fig/artifact_sample/' num2str(label) '.png'];
    %export_fig(fig_name, '-a1', '-nocrop');
end

