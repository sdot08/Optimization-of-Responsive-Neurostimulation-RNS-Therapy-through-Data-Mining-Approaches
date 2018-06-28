
%[stimulated_231, scheduled_231] = stim_date_func(Catalog_231);
%[stimulated_222, scheduled_222] = stim_date_func(Catalog_222);
scheduled = scheduled_231;
cursor = cursor_info;
inttime = cursor.Position(1) + cursor.Position(2);
if ismember(inttime, scheduled)
    scheduled = scheduled(scheduled ~= inttime);
end
    