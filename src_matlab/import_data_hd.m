%% Import data from text file.
% Script for importing data from the following text file:
%
%    /Users/hp/Desktop/summer research/ForByron_060618/NY231_2016-07-05_to_2018-06-12_hourly_20180613153300.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2018/06/20 13:26:55

%% Initialize variables.
filename = '/Users/hp/Desktop/summer research/ForByron_060618/NY231_2016-07-05_to_2018-06-12_hourly_20180613153300.csv';
delimiter = ',';
startRow = 5;

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: datetimes (%{MM/dd/yyyy HH:mm}D)
%	column4: categorical (%C)
%   column5: datetimes (%{MM/dd/yyyy HH:mm}D)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%{MM/dd/yyyy HH:mm}D%C%{MM/dd/yyyy HH:mm}D%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
h231 = table(dataArray{1:end-1}, 'VariableNames', {'patient_id','device_sn','region_start_time','timezone_region','utc_start_time','pattern_a_channel_1','pattern_a_channel_2','pattern_b_channel_1','pattern_b_channel_2','episode_starts','episode_starts_with_rx','long_episodes','magnet_swipes','saturations','hist_hours','mag_sat_hours'});

% For code requiring serial dates (datenum) instead of datetime, uncomment
% the following line(s) below to return the imported dates as datenum(s).

% NY23120160705to20180612hourly20180613153300.region_start_time=datenum(NY23120160705to20180612hourly20180613153300.region_start_time);NY23120160705to20180612hourly20180613153300.utc_start_time=datenum(NY23120160705to20180612hourly20180613153300.utc_start_time);

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;


%% Import data from text file.
% Script for importing data from the following text file:
%
%    /Users/hp/Desktop/summer research/ForByron_060618/NY231_2016-07-05_to_2018-06-12_daily_20180613153815.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2018/06/20 13:29:40

%% Initialize variables.
filename = '/Users/hp/Desktop/summer research/ForByron_060618/NY231_2016-07-05_to_2018-06-12_daily_20180613153815.csv';
delimiter = ',';
startRow = 5;

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: datetimes (%{MM/dd/yyyy HH:mm}D)
%	column4: categorical (%C)
%   column5: datetimes (%{MM/dd/yyyy HH:mm}D)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%{MM/dd/yyyy HH:mm}D%C%{MM/dd/yyyy HH:mm}D%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
d231 = table(dataArray{1:end-1}, 'VariableNames', {'patient_id','device_sn','region_start_time','timezone_region','utc_start_time','pattern_a_channel_1','pattern_a_channel_2','pattern_b_channel_1','pattern_b_channel_2','episode_starts','episode_starts_with_rx','long_episodes','magnet_swipes','saturations','hist_hours','mag_sat_hours'});

% For code requiring serial dates (datenum) instead of datetime, uncomment
% the following line(s) below to return the imported dates as datenum(s).

% NY23120160705to20180612daily20180613153815.region_start_time=datenum(NY23120160705to20180612daily20180613153815.region_start_time);NY23120160705to20180612daily20180613153815.utc_start_time=datenum(NY23120160705to20180612daily20180613153815.utc_start_time);

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;



%% Import data from text file.
% Script for importing data from the following text file:
%
%    /Users/hp/Desktop/summer research/ForByron_060618/NY222_2015-08-11_to_2018-06-12_hourly_20180613152831.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2018/06/20 13:35:10

%% Initialize variables.
filename = '/Users/hp/Desktop/summer research/ForByron_060618/NY222_2015-08-11_to_2018-06-12_hourly_20180613152831.csv';
delimiter = ',';
startRow = 5;

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: datetimes (%{MM/dd/yyyy HH:mm}D)
%	column4: categorical (%C)
%   column5: datetimes (%{MM/dd/yyyy HH:mm}D)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%{MM/dd/yyyy HH:mm}D%C%{MM/dd/yyyy HH:mm}D%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
h222 = table(dataArray{1:end-1}, 'VariableNames', {'patient_id','device_sn','region_start_time','timezone_region','utc_start_time','pattern_a_channel_1','pattern_a_channel_2','pattern_b_channel_1','pattern_b_channel_2','episode_starts','episode_starts_with_rx','long_episodes','magnet_swipes','saturations','hist_hours','mag_sat_hours'});

% For code requiring serial dates (datenum) instead of datetime, uncomment
% the following line(s) below to return the imported dates as datenum(s).

% NY22220150811to20180612hourly20180613152831.region_start_time=datenum(NY22220150811to20180612hourly20180613152831.region_start_time);NY22220150811to20180612hourly20180613152831.utc_start_time=datenum(NY22220150811to20180612hourly20180613152831.utc_start_time);

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;


%% Import data from text file.
% Script for importing data from the following text file:
%
%    /Users/hp/Desktop/summer research/ForByron_060618/NY222_2015-08-11_to_2018-06-12_daily_20180613153105.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2018/06/20 13:35:52

%% Initialize variables.
filename = '/Users/hp/Desktop/summer research/ForByron_060618/NY222_2015-08-11_to_2018-06-12_daily_20180613153105.csv';
delimiter = ',';
startRow = 5;

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: datetimes (%{MM/dd/yyyy HH:mm}D)
%	column4: categorical (%C)
%   column5: datetimes (%{MM/dd/yyyy HH:mm}D)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%{MM/dd/yyyy HH:mm}D%C%{MM/dd/yyyy HH:mm}D%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
d222 = table(dataArray{1:end-1}, 'VariableNames', {'patient_id','device_sn','region_start_time','timezone_region','utc_start_time','pattern_a_channel_1','pattern_a_channel_2','pattern_b_channel_1','pattern_b_channel_2','episode_starts','episode_starts_with_rx','long_episodes','magnet_swipes','saturations','hist_hours','mag_sat_hours'});

% For code requiring serial dates (datenum) instead of datetime, uncomment
% the following line(s) below to return the imported dates as datenum(s).

% NY22220150811to20180612daily20180613153105.region_start_time=datenum(NY22220150811to20180612daily20180613153105.region_start_time);NY22220150811to20180612daily20180613153105.utc_start_time=datenum(NY22220150811to20180612daily20180613153105.utc_start_time);

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;