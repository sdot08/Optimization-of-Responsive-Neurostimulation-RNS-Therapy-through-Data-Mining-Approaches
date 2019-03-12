%% Define file, load ECoG

subjectN = 'NY231';
filename = '131347929310220000.dat'; % Clean EEG
% filename = '131128235625610000.dat'; % With seizure activity

pth = 'R:\liua05lab\liua05labspace\RNS\FullDatasets_summer2018\231_4086548_2016-07-01_2018-12-31 EXTERNAL #PHI';

fs      = 250;

data  = readPersystDat([pth filesep filename]);

Nchan = size(data,2);
t_all = (0:size(data,1)-1)./fs;

plot_simple = 0; % 0 - only plot raw EEG; 1 - plot raw EEG with other things


%% Plot raw EEG

if(1) % Plot raw EEG
    dg_max = 1016/3;
    extm       = [];
    dat_means  = zeros(1, Nchan);
    ytk_lb     = cell(1, Nchan);

    if(plot_simple)
        figure;
    else
        f_sync = figure('OuterPosition', [0 80 1900 1000]);
        subplot(3,3,[1,2,3]);
    end
    
    for ch = 1:Nchan
        dat = data(:, ch);
        addval = (ch-1)*(dg_max+100);
        dat = dat + addval;
        extm = [extm, min(dat), max(dat)];
        dat_means(ch) = mean(dat);
        ytk_lb{ch} = ['Ch', num2str(ch)];
        hold on, plot(t_all, dat, 'k');
    end

    xlim([t_all(1) t_all(end)]);
    xlabel('Time (sec)');
    set(gca, 'YTick', dat_means, 'YTickLabel', ytk_lb);
    title([subjectN ', ' filename]);
end % Plot raw EEG


%% Calculate phase synchronization index

%define ECoG power bands of interest
% Power_band{1} = [0.5 4];     %delta, ORIGINALLY [0 4]
Power_band{1} = [4 8];       %theta
Power_band{2} = [8 12];      %alpha
Power_band{3} = [12 25];     %beta
Power_band{4} = [25 50];     %low gamma
Power_band{5} = [50 90];  %high gamma
Power_band{6} = [0.01 90]; %entire band, ORIGINALLY [0 124.9]

band_names = {'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma', 'Entire Band'};

%%%%%% Test calculating Mean Phase Coherence

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% STOP! First find out the lengths of all .dat files from each patient
% Then truncate all other .dat files to the length of the shortest file
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if (size(data,1)/fs)/20 > 2 && 0 % Do not window any more because it's easier for small number of samples to induce spurious synchronization
    windowleng = 20; % in sec
    overlap = 20; % in %
else
    windowleng = [];
    overlap = [];
end

psnyc = zeros(6, length(Power_band)-1);

for ifreq = 1:length(Power_band)-1 % Do not calculate phase synchronization for the "entire band"
    PLO = nbt_doPhaseLocking_XJedit(data, fs, Power_band{ifreq}, [], [], windowleng, overlap, [1 1]);
    
    temp = PLO.PLV';
    temp = temp(:);
    temp(isnan(temp)) = [];
    psnyc(:,ifreq) = temp; % Each row is a pair of channels, from top to bottom, the pairs are:
                           % Ch1-Ch2, Ch1-Ch3, Ch1-Ch4, Ch2-Ch3, Ch2-Ch4, Ch3-Ch4
                           
    if exist('f_sync')
        figure(f_sync);
        subplot(3,3,3+ifreq);
        colormap('jet');
        imagesc(PLO.PLV, [0 0.45]);
        
        txt = cellfun(@(x) num2str(x,3), num2cell(psnyc(:,ifreq)), 'UniformOutput',0);
        hold on, text([2,3,4,3,4,4]-0.15, [1,1,1,2,2,3], txt, 'Color','m');
        
        set(gca, 'XTick', 1:4, 'XTickLabel', {'Ch1','Ch2','Ch3','Ch4'},...
                 'YTick', 1:4, 'YTickLabel', {'Ch1','Ch2','Ch3','Ch4'});
        title([band_names{ifreq} ' [' num2str(Power_band{ifreq}(1)) ' ' num2str(Power_band{ifreq}(2)) ']Hz']);
    end
end

if(0)
    plotdir = 'H:\Personal\RNS\Results\Phase synchronization';
    fig_name = sprintf('%s%s%s_PLV_calculation_eg.png', plotdir, filesep, subjectN);
    export_fig(fig_name,'-a1','-nocrop');
end
%%%%%% Test calculating Mean Phase Coherence


%% Perform PLV simulation with randomly generated relative-phases

N = 18627; % data total number of samples
SF_num = 5000; % number of simulations to perform

PLV_sim = nan(1, SF_num);

% range of relative phase
a = -180;
b = 180;

for SF = 1:SF_num
      RP = a + (b-a)*rand(N,1); % n*phase1-m*phase2
      PLV_sim(SF) = abs(sum(exp(1i*RP)))/length(RP);
end


%% Calculate band power

data = data - repmat(mean(data,1), size(data,1), 1);

power = zeros(Nchan, length(Power_band));
for ch = 1:Nchan
    power(ch,1)  = bandpower(data(:,ch), 250, Power_band{1});
    power(ch,2)  = bandpower(data(:,ch), 250, Power_band{2});
    power(ch,3)  = bandpower(data(:,ch), 250, Power_band{3});
    power(ch,4)  = bandpower(data(:,ch), 250, Power_band{4});
    power(ch,5)  = bandpower(data(:,ch), 250, Power_band{5});
    power(ch,6)  = bandpower(data(:,ch), 250, Power_band{6});
    power(ch,7)  = bandpower(data(:,ch), 250, Power_band{7});
end

pxx = []; w = [];
for ch = 1:Nchan
    dat = data(:,ch);
    [pxx(:,ch), w] = periodogram(dat, rectwin(length(dat)), 1024, fs);
end
figure, plot(w, 10*log10(pxx));
xlim([w(1) w(end)]);
ylim([-50 50]);
xlabel('Frequency (Hz)');
ylabel('dB');
legend(ytk_lb);
