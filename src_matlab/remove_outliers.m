
%[stimulated_231, scheduled_231] = stim_date_func(Catalog_231);
%[stimulated_222, scheduled_222] = stim_date_func(Catalog_222);
%scheduled = scheduled_222;
for i = 1:length(cursor_info)
    cursor = cursor_info(i);
    inttime = cursor.Position(1) + cursor.Position(2);
    if ismember(inttime, scheduled)
        scheduled = scheduled(scheduled ~= inttime);
    end
end

plot_rmoutlier(222, scheduled, stimulated_222)
scheduled_222_ro = scheduled;
save('/Users/hp/GitHub/EEG/data/schduled_timestamp', 'scheduled_231_ro', 'scheduled_222_ro', '-v7.3');
