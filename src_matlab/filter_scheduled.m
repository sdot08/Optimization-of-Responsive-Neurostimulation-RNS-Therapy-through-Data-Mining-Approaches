function [stimulated, scheduled] = filter_scheduled(data_raw, pt_ID)
data = data_raw(:,[6, 7, end] );

prepath = '/Users/hp/GitHub/EEG/datdata/';
b = zeros(height(data), 1);
thres = 1/24/2/2;
switch pt_ID
case 222
    sched = [4/24, 16/24, 0.87];
    sched_names = {'04:00', '16:00', '20:00'};

case 231
    sched = [4/24, 10/24, 16/24, 22/24];
    sched_names = {'04:00', '10:00', '16:00', '22:00'};
end
time_masks = [];
for i = 1:length(sched)
        schedi = sched(i);
        interval = [schedi - thres, schedi + thres];
        time_masks = [time_masks, interval];
end
if pt_ID == 222
    time_masks(1) = time_masks(1) - thres;
end
disp(time_masks)
for i = 1:height(data)
    
    datai =data(i,:);
    %not real time data
    if datai.ECoGtype == 'real-time'
        b(i) = 1;
        continue
    end
    tp = datai.Timestamp_int; 
    date = floor(tp);
    time = tp - date;
    b(i) = timefilter(time, time_masks);
   
end
b = b == 1;
%stimulated = data_raw(b,:).RawLocalTimestamp;
nb = not (b);
scheduled = data_raw(nb,:).Timestamp_int;

stimulated = data_raw(b,:).Timestamp_int;



stimulated_date = floor(stimulated);
stimulated_4d = stimulated - stimulated_date;
scheduled_date = floor(scheduled);
scheduled_4d = scheduled - scheduled_date;
%length_4d = length(Catalog_Time_4d);
%x = linspace(1,length_4d, length_4d);
f_h = figure('OuterPosition', [50 50 800 600]);

scatter(stimulated_4d, stimulated_date, 'r')
hold on
scatter(scheduled_4d, scheduled_date, 'b')

set(gca, 'XTick', sched, 'XTickLabel', sched_names);
xlabel('Time during day');
ylabel('Date (MM/DD/YY)');
title(['Patient ID: ' num2str(pt_ID)])
hold off
fig_name = ['/Users/hp/GitHub/EEG/fig/' num2str(pt_ID) '_DAT_time_date.png'];
export_fig(fig_name, '-a1', '-nocrop');

