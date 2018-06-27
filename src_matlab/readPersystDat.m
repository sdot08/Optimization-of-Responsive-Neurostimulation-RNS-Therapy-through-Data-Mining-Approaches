function d = readPersystDat(datFilePath, recnum, calibration)
% read a persyst dat file (without lay file)
%
% data = readPersystDat(datFilePath, [channels])
%  returns data (time x channels)
%

if ~exist('recnum', 'var')||isempty(recnum),
    recnum = 4; % assumes 4 channels as per Neuropace RNS device
end
if ~exist('calibration', 'var')||isempty(calibration),
    calibration = 1;
end

dat_file_ID = fopen(datFilePath);
precision = 'short';
%read data from .dat file into vector of correct size, then calibrate
d = fread(dat_file_ID,[recnum,Inf],precision) * calibration;
fclose(dat_file_ID);

d = d';