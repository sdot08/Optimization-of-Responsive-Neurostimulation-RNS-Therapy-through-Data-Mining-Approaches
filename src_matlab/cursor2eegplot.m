function cursor2eegplot(data, cursor)
inttime = cursor.Position(1) + cursor.Position(2);
timestamp2eegplot(data, inttime)