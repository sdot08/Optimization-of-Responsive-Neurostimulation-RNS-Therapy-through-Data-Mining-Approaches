
%% Load in ECoG data stored in .dat files

filename = '131341699824050000.dat';
data  = readPersystDat([pth filesep filename]);
% data is a sample number X channel number matrix
% channel number is usually 4


%% Calculate power in specific frequency bands

%define ECoG power bands of interest
Power_band{1} = [0 4];        %delta
Power_band{2} = [4 8];       %theta
Power_band{3} = [8 12];      %alpha
Power_band{4} = [12 25];     %beta
Power_band{5} = [25 50];     %low gamma
Power_band{6} = [50 124.9];  %high gamma
Power_band{7} = [0 124.9];   %entire band

%code for finding spectral power in the different frequency bands.
data_each_channel_delta  = bandpower(data_each_channel_1, 250, Power_band{1});
data_each_channel_theta  = bandpower(data_each_channel_1, 250, Power_band{2});
data_each_channel_alpha  = bandpower(data_each_channel_1, 250, Power_band{3});
data_each_channel_beta  = bandpower(data_each_channel_1, 250, Power_band{4});
data_each_channel_lowgamma  = bandpower(data_each_channel_1, 250, Power_band{5});
data_each_channel_highgamma  = bandpower(data_each_channel_1, 250, Power_band{6});
data_each_channel1_all  = bandpower(data_each_channel_1, 250, Power_band{7});

