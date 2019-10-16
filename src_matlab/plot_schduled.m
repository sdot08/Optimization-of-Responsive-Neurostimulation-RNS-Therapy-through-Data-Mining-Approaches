%plot_schduled(sti_dates, sche_dates, 231)



function plot_schduled(scheduled, stimulated, all_dates, pt_ID)
stimulated_date = floor(stimulated);
stimulated_4d = stimulated - stimulated_date;
scheduled_date = floor(scheduled);
scheduled_4d = scheduled - scheduled_date;
all_dates_4d = all_dates - floor(all_dates);


stimulated_date = stimulated_date + datenum('01/01/2000', 'mm/dd/yyyy');
scheduled_date = scheduled_date + datenum('01/01/2000', 'mm/dd/yyyy');
%length_4d = length(Catalog_Time_4d);
%x = linspace(1,length_4d, length_4d);
%f_h = figure('OuterPosition', [50 50 800 600]);


switch pt_ID
case 222
    sched = [4/24, 16/24, 0.87];
    sched_names = {'04:00', '16:00', '20:00'};

case 231
    sched = [4/24, 10/24, 16/24, 22/24];
    sched_names = {'4:00', '10:00', '16:00', '22:00'};
    
case 201 
    sched = [(16+5/6)/24, (17+5/6)/24];
    sched_names = {'16:50', '17:50'};
case 229
    sched = [(9+5/6)/24, (10+5/6)/24, (22+5/6)/24];
    sched_names = {'09:50', '10:50', '22:50'};
    horizontal_lines_vals = [date2int('10/09/2017'),date2int('02/13/2019')];
case 226
    sched = [(20+5/6)/24, (21+4/6)/24, (22+4/6)/24];
    sched_names = {'20:50','21:40', '22:40'};
case 241
    sched = [(10+4/6)/24, (22+4/6)/24];
    sched_names = {'10:40', '22:40'};

case 251
    sched = [ 2/24, (8+3/6)/24,(14+3/6)/24, (20+3/6)/24];
    sched_names = {'02:30', '08:30',  '14:30','20:30', };
    horizontal_lines_vals = [date2int('08/26/2018'),date2int('04/28/2019')];
case 239
    sched = [(8+3/6)/24, (20+3/6)/24];
    sched_names = {'8:30', '20:30'};
    % horizontal_lines_vals = [date2int('07/13/2018'),date2int('01/31/2019'), date2int('07/11/2018'),date2int('01/04/2018')];
    horizontal_lines_vals = [date2int('01/04/2018'),date2int('05/24/2019')];

case 225
    sched = [(10+5/6)/24, (22+5/6)/24];
    sched_names = {'10:50', '22:50'};
    %     horizontal_lines_vals = [date2int('08/24/2016'),date2int('04/12/2017'), date2int('10/13/2017'),date2int('06/05/2018')];
    horizontal_lines_vals = [date2int('07/19/2016'),date2int('07/12/2017')];
case 217
    sched = [ 3/24, 9/24,13/24, (20+3/6)/24];
    sched_names = {'03:00', '09:00',  '13:00','20:40', };    
    horizontal_lines_vals = [date2int('07/19/2018'),date2int('05/18/2019')];
end
sched = [0,sched,1];
sched_names = ['00:00',sched_names,'24:00'];
%sched_names = ['1','2','3','4','5','6'];

figure('OuterPosition', [80 20 2200 1800]);
scatter(scheduled_4d, scheduled_date, [], [0,0.5,1])
hold on
scatter(stimulated_4d, stimulated_date, 'm')
hold on

for i = 1:length(horizontal_lines_vals)
    hold on
    line(all_dates_4d, horizontal_lines_vals(i)*ones(size(all_dates_4d)),'Color', 'green', 'LineWidth', 3);
end

set(gca, 'XTick', sched, 'XTickLabel', sched_names);
ytks = get(gca, 'YTick');
set(gca, 'YTickLabel', datestr(ytks), 'FontSize',26, 'FontWeight','bold');

xlabel('Time during day');
ylabel('Date (DD/MM/YY)');
legend('Other(Long episode, Magnet, etc. )', 'Scheduled','Clean periods');
title(['Patient ', num2str(pt_ID)]);
%title(['Patient ID: ' num2str(pt_ID) '    Time vs Date for EEG segments'])
hold off
fig_name = ['/Users/hp/GitHub/EEG/fig/scheduled/' num2str(pt_ID) '_time_date.png'];
export_fig(fig_name, '-a1', '-nocrop');