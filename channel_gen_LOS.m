function [pcsi, ecsi, AoA, AoD, alpha] = channel_gen_LOS(pnr_dB,ITER, Num_paths, Lest,Nrf, Nt, Nr)
%------------------------System Parameters---------------------------------
Num_BS_Antennas=Nt;
BSAntennas_Index=0:1:Num_BS_Antennas-1;
Num_BS_RFchains=Nrf;
Num_MS_Antennas=Nr;
MSAntennas_Index=0:1:Num_MS_Antennas-1;
Num_MS_RFchains=Nrf;
Num_Qbits=7;

%---------------------- Channel Parameters -------------------------------
Num_paths_est=Lest;
Carrier_Freq=28*10^9;
lambda=3*10^8/Carrier_Freq;
n_pathloss=3;
Tx_Rx_dist=50;
ro=((lambda/(4*pi*5))^2)*(5/Tx_Rx_dist)^n_pathloss;
Pt_avg=10^(.7);
Pr_avg=Pt_avg*ro;

%---------------- Channel Estimation Algorithm Parameters------------------
G_BS = Num_paths_est * 256;
G_MS = Num_paths_est * 256;
K_BS=2;
K_MS=2;
S=floor(max(log(G_BS/Num_paths_est)/log(K_BS),log(G_MS/Num_paths_est)/log(K_MS)));

Pr_alloc=power_allocation(Num_BS_Antennas,Num_BS_RFchains,BSAntennas_Index,G_BS,G_MS,K_BS,K_MS,Num_paths_est,Num_Qbits);
Pr=abs(Pr_avg*Pr_alloc*S);

sin_theta_grid_BS = linspace(-1, 1, G_BS);
sin_theta_grid_MS = linspace(-1, 1, G_MS);

AbG = zeros(Num_BS_Antennas, G_BS);
for g=1:1:G_BS

    AbG(:,g)=sqrt(1/Num_BS_Antennas)*exp(1j*pi*sin_theta_grid_BS(g)*BSAntennas_Index.');
end

% Am generation
AmG = zeros(Num_MS_Antennas, G_MS);
for g=1:1:G_MS
    AmG(:,g)=sqrt(1/Num_MS_Antennas)*exp(1j*pi*sin_theta_grid_MS(g)*MSAntennas_Index.');
end
%--------------------------------------------------------------------------
ecsi = zeros(ITER,Num_MS_Antennas,Num_BS_Antennas);
error_NMSE = zeros(ITER,1);
pcsi = zeros(ITER,Num_MS_Antennas,Num_BS_Antennas);
AoA_all = zeros(ITER, Num_paths);
AoD_all = zeros(ITER, Num_paths);
alpha_all = zeros(ITER, Num_paths); 

for iter=1:1:ITER % Vòng lặp chính, dựa trên bài báo của Alkhateeb)
    if mod(iter,100)==0 
        iter 
    end

    % Channel Generation
    AoD = pi*rand(1,Num_paths) - pi/2;
    AoA = pi*rand(1,Num_paths) - pi/2;

    alpha(1) = 1;
    alpha(2:Num_paths) = sqrt(10^(-5)) * (randn(1,Num_paths-1)+1i*randn(1,Num_paths-1))/sqrt(2);

    AoA_all(iter, :) = AoA;
    AoD_all(iter, :) = AoD;
    alpha_all(iter, :) = alpha;
    % Channel construction
    Channel=zeros(Num_MS_Antennas,Num_BS_Antennas);
    for l=1:1:Num_paths
        Abh(:,l)=sqrt(1/Num_BS_Antennas)*exp(1j*pi*sin(AoD(l))*BSAntennas_Index.'); % a_BS(φ)
        Amh(:,l)=sqrt(1/Num_MS_Antennas)*exp(1j*pi*sin(AoA(l))*MSAntennas_Index.'); % a_MS(θ)

        Channel = Channel + sqrt(Num_BS_Antennas*Num_MS_Antennas / Num_paths) * alpha(l) * Amh(:,l) * Abh(:,l)';

    end
    pcsi(iter,:,:) = reshape(Channel, [1, Num_MS_Antennas, Num_BS_Antennas]);
    
    pnr=10^(0.1*pnr_dB);
    No=Pr_avg/pnr;

    % Khởi tạo tham số cho thuật toán
    KB_final=[]; KM_final=[];
    yv_final_measurement = [];
    W_final_measurement = [];
    F_final_measurement = [];
    KB_hist = zeros(Num_paths_est, S);
    KM_hist = zeros(Num_paths_est, S);

    for l=1:1:Num_paths_est % Số lần lặp

        KB_star=1:1:K_BS*Num_paths_est;
        KM_star=1:1:K_MS*Num_paths_est;
 
        for t=1:1:S

            G_matrix_BS=zeros(K_BS*Num_paths_est,G_BS);

            Block_size_BS=G_BS/(Num_paths_est*K_BS^t); 
            Block_BS=[ones(1,Block_size_BS)];

            for k=1:1:K_BS*Num_paths_est
                G_matrix_BS(k,(KB_star(k)-1)*Block_size_BS+1:(KB_star(k))*Block_size_BS)=Block_BS;
            end

            G_matrix_MS=zeros(K_MS*Num_paths_est,G_MS);
            Block_size_MS=G_MS/(Num_paths_est*K_MS^t);
            Block_MS=[ones(1,Block_size_MS)];

            for k=1:1:K_MS*Num_paths_est
                G_matrix_MS(k,(KM_star(k)-1)*Block_size_MS+1:(KM_star(k))*Block_size_MS)=Block_MS;
            end

            F_UC=(AbG*AbG')^(-1)*(AbG)*G_matrix_BS';
            W_UC=(AmG*AmG')^(-1)*(AmG)*G_matrix_MS';

            F_UC=F_UC*diag(1./sqrt(diag(F_UC'*F_UC)));
            W_UC=W_UC*diag(1./sqrt(diag(W_UC'*W_UC)));

            F_HP = zeros(Num_BS_Antennas, K_BS*Num_paths_est);
            W_HP = zeros(Num_MS_Antennas, K_MS*Num_paths_est);

            for m=1:1:K_BS*Num_paths_est
                [F_HP(:,m)]=HybridPrecoding(F_UC(:,m),Num_BS_Antennas,Num_BS_RFchains,Num_Qbits);
            end

            for n=1:1:K_MS*Num_paths_est
                [W_HP(:,n)]=HybridPrecoding(W_UC(:,n),Num_MS_Antennas,Num_MS_RFchains,Num_Qbits);
            end
            
            Noise=W_HP'*(sqrt(No/2)*(randn(Num_MS_Antennas,K_BS*Num_paths_est)+1j*randn(Num_MS_Antennas,K_BS*Num_paths_est)));

            Y=sqrt(Pr(t))*W_HP'*Channel*F_HP+Noise;

            yv_raw = reshape(Y,K_BS*K_MS*Num_paths_est^2,1);    

            
            if(t==S && l==1)
                yv_final_measurement = yv_raw / sqrt(Pr(t));
                W_final_measurement = W_HP;
                F_final_measurement = F_HP;
            end
            
            
            yv_residual = yv_raw;
            for i=1:1:length(KB_final)
                A1=transpose(F_HP)*conj(AbG(:,KB_final(i)+1));
                A2=W_HP'*AmG(:,KM_final(i)+1);
                Prev_path_cont=kron(A1,A2);
                denominator = Prev_path_cont' * Prev_path_cont;
                if abs(denominator) > 1e-10
                    Alp = (yv_residual' * Prev_path_cont) / denominator;
                    yv_residual = yv_residual - Alp*Prev_path_cont;
                end
            end
            
            Y_residual = reshape(yv_residual,K_MS*Num_paths_est,K_BS*Num_paths_est);
            [val, ~]=max(abs(Y_residual));
            Max=max(val);
            [KM_temp, KB_temp]=find(abs(Y_residual)==Max);
            
            if isempty(KM_temp) 
                KM_temp=1; 
                KB_temp=1; 
            end % Kiểm tra

            KB_hist(l,t)=KB_star(KB_temp(1));
            KM_hist(l,t)=KM_star(KM_temp(1));
            
            if(t==S)
                KB_final=[KB_final KB_star(KB_temp(1))-1];
                KM_final=[KM_final KM_star(KM_temp(1))-1];
            end
            
            for ln=1:1:l
                KB_star((ln-1)*K_BS+1:ln*K_BS)=(KB_hist(ln,t)-1)*K_BS+1:1:(KB_hist(ln,t))*K_BS;
                KM_star((ln-1)*K_MS+1:ln*K_MS)=(KM_hist(ln,t)-1)*K_MS+1:1:(KM_hist(ln,t))*K_MS;
            end
        end 
    end 
    
    % --- Ước tính Độ lợi kênh (alpha) ---
    Epsix = []; 
    for l_path=1:1:Num_paths_est
        A1 = transpose(F_final_measurement) * conj(AbG(:,KB_final(l_path)+1));
        A2 = W_final_measurement' * AmG(:,KM_final(l_path)+1);
        E_col = kron(A1,A2);
        Epsix = [Epsix E_col];
    end

    I = eye(size(Epsix, 2));
    alpha_est = (Epsix' * Epsix + lambda * I) \ (Epsix' * yv_final_measurement);
    
    % --- Tái tạo Kênh ---
    Channel_est=zeros(Num_MS_Antennas,Num_BS_Antennas);
    
    sin_AoD_est = sin_theta_grid_BS(KB_final + 1);
    sin_AoA_est = sin_theta_grid_MS(KM_final + 1);

    for l=1:1:Num_paths_est
        % Tái tạo vector lái tia bằng chính giá trị sin() đã tra cứu
        Abh_est(:,l) = sqrt(1/Num_BS_Antennas)*exp(1j*pi*sin_AoD_est(l)*BSAntennas_Index.');
        Amh_est(:,l) = sqrt(1/Num_MS_Antennas)*exp(1j*pi*sin_AoA_est(l)*MSAntennas_Index.');
        
        Channel_est = Channel_est + alpha_est(l) * Amh_est(:,l) * Abh_est(:,l)';
    end
    
    ecsi(iter,:,:) = reshape(Channel_est, [1, Num_MS_Antennas, Num_BS_Antennas]);
    error_NMSE(iter)=(norm(Channel_est-Channel,'fro')/norm(Channel,'fro'))^2;
end

AoA = AoA_all;
AoD = AoD_all;
alpha = alpha_all;

disp('Final Average NMSE:')
mean(error_NMSE)
end