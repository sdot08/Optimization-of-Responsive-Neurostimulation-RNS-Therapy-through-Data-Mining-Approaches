function flag = timefilter(time, masks)
flag = 1;
for i = 1:2:length(masks)
    if time < masks(i + 1) && time > masks(i)
        flag = 0;
    end
end

