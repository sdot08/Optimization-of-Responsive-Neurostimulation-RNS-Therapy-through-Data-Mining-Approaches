clear all
import_data;
% convert convert the value in column 'RawLocalTimestamp' from str to
% integer
pt_ID = 222;
Catalog = preprocess_time2int(Catalog_raw, 'RawLocalTimestamp', pt_ID);
Catalog_Time = Catalog.Timestamp_int;
Catalog_Time_date = floor(Catalog_Time) - 6030;
Catalog_Time_4d = Catalog_Time - Catalog_Time_date - 6030;

%length_4d = length(Catalog_Time_4d);
%x = linspace(1,length_4d, length_4d);
f_h = figure('OuterPosition', [50 50 800 600]);
scatter(Catalog_Time_4d, Catalog_Time_date)

switch pt_ID
    case 222
        sched = [4/24, 16/24, 20/24];
        sched_names = {'04:00', '16:00', '20:00'};
        
    case 231
        sched = [4/24, 10/24, 16/24, 22/24];
        sched_names = {'04:00', '10:00', '16:00', '22:00'};
        
end

set(gca, 'XTick', sched, 'XTickLabel', sched_names);
xlabel('Time during day');
ylabel('Date (MM/DD/YY)');
title(['Patient ID: ' num2str(pt_ID)])

fig_name = ['/Users/hp/Desktop/summer research/ForByron_060618/fig/' num2str(pt_ID) '_DAT_time_date.png'];
export_fig(fig_name, '-a1', '-nocrop');