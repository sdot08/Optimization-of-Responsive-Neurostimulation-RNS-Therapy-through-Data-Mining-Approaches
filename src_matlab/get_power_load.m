%% Calculate power in specific frequency bands
%% input path of the file
%% output the 28 power features corrsponds to the file
function channel_power = get_power_load(data, if_le, high_cut)


data  = readPersystDat('/Users/hp/GitHub/EEG/datdata/231/130901185189430000.dat');


%define ECoG power bands of interest
if high_cut >= 124
    high_cut = 124.9;
end
Power_band{1} = [0.5 4];        %delta
Power_band{2} = [4 8];       %theta
Power_band{3} = [8 12];      %alpha
Power_band{4} = [12 25];     %beta
Power_band{5} = [25 50];     %low gamma
Power_band{6} = [50 high_cut];  %high gamma
Power_band{7} = [0.1 high_cut];   %entire band


data_each_channel_delta = zeros(4,1, 'double');
data_each_channel_theta = zeros(4,1, 'double');
data_each_channel_alpha = zeros(4,1, 'double');
data_each_channel_beta = zeros(4,1, 'double');
data_each_channel_lowgamma = zeros(4,1, 'double');
data_each_channel_highgamma = zeros(4,1, 'double');
data_each_channel_all = zeros(4,1, 'double');
channel_power = zeros(64,1, 'double');

for i = 1:4
    data_each_channel = data(:,i);
    
    data_each_channel = data_each_channel - mean(data_each_channel);
    
    %code for finding spectral power in the different frequency bands.
    if if_le
        data_each_channel_delta(i)  = bandpower_byron(data_each_channel, 250, Power_band{1});
        data_each_channel_theta(i)  = bandpower_byron(data_each_channel, 250, Power_band{2});
        data_each_channel_alpha(i)  = bandpower_byron(data_each_channel, 250, Power_band{3});
        data_each_channel_beta(i)  = bandpower_byron(data_each_channel, 250, Power_band{4});
        data_each_channel_lowgamma(i)  = bandpower_byron(data_each_channel, 250, Power_band{5});
        data_each_channel_highgamma(i)  = bandpower_byron(data_each_channel, 250, Power_band{6});
        data_each_channel_all(i)  = bandpower_byron(data_each_channel, 250, Power_band{7});
    else
        data_each_channel_delta(i)  = bandpower(data_each_channel, 250, Power_band{1});
        data_each_channel_theta(i)  = bandpower(data_each_channel, 250, Power_band{2});
        data_each_channel_alpha(i)  = bandpower(data_each_channel, 250, Power_band{3});
        data_each_channel_beta(i)  = bandpower(data_each_channel, 250, Power_band{4});
        data_each_channel_lowgamma(i)  = bandpower(data_each_channel, 250, Power_band{5});
        data_each_channel_highgamma(i)  = bandpower(data_each_channel, 250, Power_band{6});
        data_each_channel_all(i)  = bandpower(data_each_channel, 250, Power_band{7});
        end
end
    channel_power(1:4) = data_each_channel_delta;
    channel_power(5:8) = data_each_channel_theta;
    channel_power(9:12) = data_each_channel_alpha;
    channel_power(13:16) = data_each_channel_beta;
    channel_power(17:20) = data_each_channel_lowgamma;
    channel_power(21:24) = data_each_channel_highgamma;
    channel_power(25:28) = data_each_channel_all;

%%%%%% Test calculating Mean Phase Coherence

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% STOP! First find out the lengths of all .dat files from each patient
% Then truncate all other .dat files to the length of the shortest file
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

data = data(1:15513, :);
fs = 250;

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
                           
end
%%%%%% Test calculating Mean Phase Coherence


