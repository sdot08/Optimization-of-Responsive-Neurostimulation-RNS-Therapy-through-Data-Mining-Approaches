%% check if the sum of the count of four channels equals the column episode_starts
%% input: table t
%% output: difference and the histogram plot (or the bool that indicated if they are the same)
function diff = checksum(t)
diff = t.pattern_a_channel_1 + t.pattern_a_channel_2 + t.pattern_b_channel_1 + t.pattern_b_channel_2 - t.episode_starts
histogram(diff)
bool =  (t.pattern_a_channel_1 + t.pattern_a_channel_2 + t.pattern_b_channel_1 + t.pattern_b_channel_2 == t.episode_starts) 
