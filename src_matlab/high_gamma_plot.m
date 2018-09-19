prepath = '/Users/hp/GitHub/EEG/datdata/';
filename = '131558828283370000.dat';
high_gamma = '31.875';
label = 'bad';
sti = 'sti';
path = strcat(prepath, filename);

label = ['/' label '_' sti '/' label '_' sti '_' high_gamma '_' filename(1:end - 4)];
stim_detection_c(path, 1, label)