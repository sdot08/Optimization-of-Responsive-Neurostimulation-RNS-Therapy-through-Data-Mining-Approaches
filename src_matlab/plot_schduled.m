%plot_schduled(sti_dates, sche_dates, 231)



function plot_schduled(scheduled, stimulated, pt_ID)
stimulated_date = floor(stimulated);
stimulated_4d = stimulated - stimulated_date;
scheduled_date = floor(scheduled);
scheduled_4d = scheduled - scheduled_date;

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
    sched_names = {'04:00', '10:00', '16:00', '22:00'};
    
case 201 
    sched = [(16+5/6)/24, (17+5/6)/24];
    sched_names = {'16:50', '17:50'};
case 229
    sched = [(9+5/6)/24, (10+5/6)/24, (22+5/6)/24];
    sched_names = {'09:50', '10:50', '22:50'};
case 226
    sched = [(20+5/6)/24, (21+4/6)/24, (22+4/6)/24];
    sched_names = {'20:50','21:40', '22:40'};
case 241
        sched = [(10+4/6)/24, (22+4/6)/24];
        sched_names = {'10:40', '22:40'};
    
end
sched = [0,sched,1];
sched_names = ['00:00',sched_names,'24:00'];


figure('OuterPosition', [80 20 2200 1800]);
scatter(scheduled_4d, scheduled_date, 'b')
hold on
scatter(stimulated_4d, stimulated_date, 'r')

set(gca, 'XTick', sched, 'XTickLabel', sched_names);
ytks = get(gca, 'YTick');
set(gca, 'YTickLabel', datestr(ytks), 'FontSize',26, 'FontWeight','bold');

xlabel('Time during day');
ylabel('Date (DD/MM/YY)');
legend('Scheduled','Other(Long episode, Magnet, etc. )')
title(['Patient ID: ' num2str(pt_ID) '    Time vs Date for EEG segments'])
hold off
fig_name = ['/Users/hp/GitHub/EEG/fig/scheduled/' num2str(pt_ID) '_time_date.png'];
export_fig(fig_name, '-a1', '-nocrop');