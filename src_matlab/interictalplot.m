

%10 - 100 Hz band pass filter (Donoho and Johnstone, 1994)
%8th order (4 poles at each cut off frequency) butterworth IIS
%Filter coeffs: Filter constructed using FDAtool

SOS_10_100 = [1,0,-1,1,1.33005388381822,0.651526241334396;1,0,-1,1,-1.77694660164778,0.835830280197904;1,0,-1,1,-1.55074915261902,0.609131909820269;1,0,-1,1,0.998310234198153,0.276325818973493];
G_10_100 = [0.795158533814479;0.795158533814479;0.691562244434348;0.691562244434348;1];
path = '/Users/hp/GitHub/EEG/datdata/131124920161910000.dat';
data  = readPersystDat(path);
%code for finding spikes. For computing spike rate divide by length of
%data. 
%data_each_channel_donoho = filtfilt(SOS_10_100, G_10_100, data_each_channel);  %applying band pass filter to data
%IIS_threshold = 7.5*((median(abs(data_each_channel_donoho)))/0.6745);          %See Donoho and Johnstone, 1994
%[pks,locp] = findpeaks(abs(data_each_channel_donoho), 'MINPEAKHEIGHT', IIS_threshold, 'MINPEAKDISTANCE',10);


fs = 250;
t = (0:size(data,1)-1)/fs; %compute the time of each point in EEG wave

dg_max = 1016;  %graph stuff

f_stim = figure;

Nchan = size(data,2); % Number of channels
extm       = []; %graph stuff
dat_means  = zeros(1, Nchan);
ytk_lb     = cell(1, Nchan); %graph stuff channel name holder
idx_ch_all = cell(1, Nchan); %index of the start of the stimulation for 4 channels
for ch = 1:Nchan
    data_each_channel = data(:,ch);
    %code for finding spikes. For computing spike rate divide by length of
    %data. 
    data_each_channel_donoho = filtfilt(SOS_10_100, G_10_100, data_each_channel);  %applying band pass filter to data
    IIS_threshold = 7.5*((median(abs(data_each_channel_donoho)))/0.6745);          %See Donoho and Johnstone, 1994
    [pks,locp] = findpeaks(abs(data_each_channel_donoho), 'MINPEAKHEIGHT', IIS_threshold, 'MINPEAKDISTANCE',30);
    %disp('ch=')
    %disp(ch)
    %disp(pks)
    %disp(locp)
    figure(f_stim);
    dat = data_each_channel_donoho;
    addval = (ch-1)*(dg_max+100);
    dat = dat + addval;
    extm = [extm, min(dat), max(dat)];
    dat_means(ch) = mean(dat);
    ytk_lb{ch} = ['Ch', num2str(ch)];
    hold on, plot(t, dat, 'k'); 
    hold on, plot(t(locp), dat(locp), 'ro','MarkerSize',6);
    end

figure(f_stim);
xlim([t(1) t(end)]);
xlabel('Time (sec)');
%set(gca, 'YTick', dat_means, 'YTickLabel', ytk_lb);
%title(['Patient 231, Datatime: ' num2str(label)])
%fig_name = ['/Users/hp/Desktop/summer research/ForByron_060618/fig/artifact_sample/' num2str(label) '.png'];
%export_fig(fig_name, '-a1', '-nocrop');

