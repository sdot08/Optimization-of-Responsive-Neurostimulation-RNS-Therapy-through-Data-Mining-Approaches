%%plot EEG wave with stimulation circled for four files
prepath = '../datdata/';
%131720352867140000.dat for patient 231 at time 05/27/2018 21:55:02, with
%simulation
%131720352682110000.dat for patient 231 at time 05/27/2018 15:55:13,
%without simulation
%131717740021760000.dat for patient 231 at time 5/25/2018  3:55:40 PM
%131707393222130000.dat for patient 231 at time 5/13/2018  2:59:04 PM
filename1 = '131720352867140000.dat';
filename2 = '131720352682110000.dat';
filename3 = '131717740021760000.dat';
filename4 = '131707393222130000.dat';
label1 = '05.27.2018 21:55:02';
label2 = '05.27.2018 15:55:13';
label3 = '5.25.2018  3:55:40';
label4 = '5.13.2018  2:59:04';
path = strcat(prepath, filename4);
stim_detection(path, 1 , label4)
