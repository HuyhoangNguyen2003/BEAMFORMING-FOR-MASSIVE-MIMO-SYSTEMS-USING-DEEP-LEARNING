pnr_dB = 10;
ITER = 10000;
Num_paths = 3;
Lest = 3;
Nrf = 1;

Nt = 64;
Nr = 8;

% generate pcsi and ecsi.   
% pcsi: perfect csi ecsi:estimated csi
[pcsi, ecsi] = channel_gen_LOS_test(pnr_dB, ITER, Num_paths, Lest, Nrf, Nt, Nr);

save('pcsi.mat', 'pcsi')
save('ecsi.mat', 'ecsi')

% save('AoA.mat', 'AoA')
% save('AoD.mat', 'AoD')
% save('alpha.mat', 'alpha')

% xuat ra file excel (neu can) 

% pcsi_excel = 'pcsi.xlsx';
% writematrix(pcsi, pcsi_excel);

% ecsi_excel = 'ecsi.xlsx';
% writematrix(ecsi, ecsi_excel);