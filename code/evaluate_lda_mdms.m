function evaluate_lda_mdms(config)

for dataset_ind = 1:length(config.dataset_names)
    %%%%%%%%%%%%%%%%%%%%
    % set local config %
    %%%%%%%%%%%%%%%%%%%%
    data_dir = [config.data_dir, '\', config.dataset_names{dataset_ind}];
    code_dir = config.code_dir;
    save_dir = config.save_dir;
    sub_num = config.sub_num(dataset_ind);
    nb_senator = config.nb_senator(dataset_ind);
    mov_num = config.mov_num(dataset_ind);
    trial_num = config.trial_num(dataset_ind);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load features, labels, and optimized parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cd(data_dir);
    load(['F_c.mat']);
    cd(code_dir);
    feat_dim = size(F{1,1,1},2);

    %%%%%%%%%%
    % buffer %
    %%%%%%%%%%
    acc_lda = zeros(1, sub_num);
    pred_lda = cell(1, sub_num);
    local_z_mu = zeros(sub_num, feat_dim);
    local_z_sigma = zeros(sub_num, feat_dim);
    LDA_Sigmas = zeros(sub_num, feat_dim, feat_dim);
    LDA_Mus = zeros(sub_num, mov_num, feat_dim);
    weight_lib = zeros(sub_num, sub_num-1);
    senator_index_lib = cell(1,sub_num);

    %%%%%%%%%%%%%%%%%
    % train 25 LDAs %
    %%%%%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
        data = []; label = [];
        for trial_ind = 1:trial_num
            for mov_ind = 1:mov_num
                data = [data; F{sub_ind, trial_ind, mov_ind}];
                label = [label; c{sub_ind, trial_ind, mov_ind}];
            end
        end
    
        [ZF, local_z_mu(sub_ind,:), local_z_sigma(sub_ind,:)] = zscore(data);
        LDA =  fitcdiscr(ZF, label, 'DiscrimType', 'pseudoLinear');
        LDA_Sigmas(sub_ind,:,:) = LDA.Sigma;
        LDA_Mus(sub_ind,:,:) = LDA.Mu;
    end
    
    disp(['lda mdms dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' LDAs done'])

    %%%%%%%%%%%%%%
    % evaluation %
    %%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
        
        % preparation
        sub_ind_seq = 1:sub_num;
        sub_ind_seq(sub_ind) = [];
        S_cal = []; L_cal = [];
        S_tes = []; L_tes = [];
        
        if dataset_ind == 1
            for mov_ind = 1:mov_num
                S_cal = [S_cal; F{sub_ind, 1, mov_ind}]; % 1st trial
                L_cal = [L_cal; c{sub_ind, 1, mov_ind}];
            end
        
            for trial_ind = 3:5
               for mov_ind = 1:mov_num
                    S_tes = [S_tes; F{sub_ind, trial_ind, mov_ind}]; % 3rd to 5th trial
                    L_tes = [L_tes; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        else
            for trial_ind = 1:2
                for mov_ind = 1:mov_num
                    S_cal = [S_cal; F{sub_ind, trial_ind, mov_ind}]; % 1st and 2nd trials
                    L_cal = [L_cal; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        
            for trial_ind = 5:6
                for mov_ind = 1:mov_num
                    S_tes = [S_tes; F{sub_ind, trial_ind, mov_ind}]; % 5th and 6th trials
                    L_tes = [L_tes; c{sub_ind, trial_ind, mov_ind}];
                end
            end 
        end
            
        candidate_LDA_Mus = LDA_Mus;
        candidate_LDA_Mus(sub_ind,:,:) = [];        
        candidate_LDA_Sigmas = LDA_Sigmas;
        candidate_LDA_Sigmas(sub_ind,:,:) = [];    
    
        z_mu_dammy = local_z_mu; z_sigma_dammy = local_z_sigma;
        z_mu_dammy(sub_ind,:) = []; z_sigma_dammy(sub_ind,:) = [];   
    
        % select senators besed on MDMs
        est_XPs = zeros(1, sub_num-1);
     
        for i = 1:sub_num-1  
            MDMs = zeros(1, config.mdms_num);
        
            % train tranferability predictive model
            MDMs_train = zeros(sub_num-2, config.mdms_num);
            XPs_train = zeros(1, sub_num-2);
        
            sub_source = [];
            for trial_ind = 1:trial_num
                for mov_ind = 1:mov_num
                    sub_source = [sub_source; F{sub_ind_seq(i), trial_ind, mov_ind}];
                end
            end
            sub_source = (sub_source - z_mu_dammy(i,:)) ./ z_sigma_dammy(i,:);
            median_sub_source = median(sub_source);
            hist_sub_source = hist(median_sub_source, length(median_sub_source))/size(median_sub_source,2);
            hist_sub_source = hist_sub_source + eps;
            
            sub_ind_seq_dammy = sub_ind_seq;
            sub_ind_seq_dammy(i) = [];
            
            for j = 1:sub_num-2
                another_source = []; another_source_L = [];
                for trial_ind = 1:trial_num
                    for mov_ind = 1:mov_num
                        another_source = [another_source; F{sub_ind_seq_dammy(j), trial_ind, mov_ind}];
                        another_source_L = [another_source_L; c{sub_ind_seq_dammy(j), trial_ind, mov_ind}];
                    end
                end
                
                another_source = (another_source - z_mu_dammy(i,:)) ./ z_sigma_dammy(i,:);
            
                median_another_source = median(another_source);
                hist_another_source = hist(median_another_source, length(median_another_source))/size(median_another_source,2);
                hist_another_source = hist_another_source + eps;
                       
                % Euclidian distance
                MDMs_train(j, 1) = sum((hist_sub_source - hist_another_source).^2);
        
                % Correlation distance
                temp = corrcoef(hist_sub_source, hist_another_source);
                MDMs_train(j, 2) = temp(1,2);
        
                % Chebyshev distance
                MDMs_train(j, 3) = max(abs(hist_sub_source - hist_another_source));
        
                % Cosine distance
                MDMs_train(j, 4) = 1 - (sum(hist_sub_source.*hist_another_source) / (sqrt(sum(hist_sub_source.^2))*(sqrt(sum(hist_another_source.^2)))));
                                         
                % Kullback-Leibler divergence
                KL1 = sum(hist_sub_source .* (log2(hist_sub_source)-log2(hist_another_source)));
                KL2 = sum(hist_another_source .* (log2(hist_another_source)-log2(hist_sub_source)));
                MDMs_train(j, 5) = (KL1+KL2)/2;      
            
                % Cross-subjecct model performance
                prob = zeros(length(another_source_L),mov_num);
                Sigma = squeeze(candidate_LDA_Sigmas(i,:,:));
                Mu = squeeze(candidate_LDA_Mus(i,:,:));
                for k = 1:max(another_source_L)
                    prob(:,k) = another_source*pinv(Sigma')*Mu(k,:)' - (1/2)*Mu(k,:)*pinv(Sigma')*Mu(k,:)' - log(2);
                end
            
                [~, pred_temp] = max(prob');
                Acc = sum(pred_temp'==another_source_L)/length(another_source_L);             
                XPs_train(j) = Acc;
            end
        
            % linear epcilon SVR
            cmd = ['-q -s 3 -t 0 -c ', num2str(10^1)];
            [Z_MDMs_train, MDMs_mu, MDMs_sigma] = zscore(MDMs_train);
            SVR = svmtrain(XPs_train', Z_MDMs_train, cmd);
        
            % estimate transferability for target data
            S_cal_dammy = S_cal;
            S_cal_dammy = (S_cal_dammy - z_mu_dammy(i,:)) ./ z_sigma_dammy(i,:);
            median_cal = median(S_cal_dammy);                                   % calculate median across windows 
            hist_cal = hist(median_cal, length(median_cal))/size(median_cal,2); % probability density function (sum equal to 1)
            hist_cal = hist_cal + eps;
        
            temp = corrcoef(hist_sub_source, hist_cal);
            KL1 = sum(hist_sub_source .* (log2(hist_cal)-log2(hist_cal)));
            KL2 = sum(hist_cal .* (log2(hist_cal)-log2(hist_sub_source)));
        
            MDMs(1) = sum((hist_sub_source - hist_cal).^2);
            MDMs(2) = temp(1,2);
            MDMs(3) = max(abs(hist_sub_source - hist_cal));
            MDMs(4) = 1 - (sum(hist_sub_source.*hist_cal) / (sqrt(sum(hist_sub_source.^2))*(sqrt(sum(hist_cal.^2)))));                                 
            MDMs(5) = (KL1+KL2)/2;                                         
        
            % estimate transferability
            Z_MDMs = (MDMs- MDMs_mu) ./ MDMs_sigma;
            est_XPs(i) = svmpredict(1, Z_MDMs, SVR,'-q');
        end
        weight_lib(sub_ind, :) = est_XPs;
    
        % select senator LDAs
        [~, b] = sort(est_XPs,'descend');
        senator_index = b(1:nb_senator);
        senator_index_lib{sub_ind} = senator_index; 
        naive_prob = zeros(length(L_tes), mov_num);
        
        for nb_ind = 1:nb_senator
            % normalization
            S_tes_dammy = (S_tes - local_z_mu(sub_ind_seq(senator_index(nb_ind)), :)) ./ local_z_sigma(sub_ind_seq(senator_index(nb_ind)), :);
                    
            % naive performance
            naive_mu = squeeze(LDA_Mus(sub_ind_seq(senator_index(nb_ind)),:,:));
            naive_sigma = squeeze(LDA_Sigmas(sub_ind_seq(senator_index(nb_ind)),:,:));
        
            for j = 1:mov_num
                temp_prob = S_tes_dammy*pinv(naive_sigma')*naive_mu(j,:)' - (1/2)*naive_mu(j,:)*pinv(naive_sigma')*naive_mu(j,:)' - log(2);
                naive_prob(:, j) = naive_prob(:, j) + temp_prob;
            end   
        end
        
        % recognition 
        [~, pred_naive_temp] = max(naive_prob');
        pred_lda{sub_ind} = pred_naive_temp;
        acc_lda(sub_ind) = sum(pred_naive_temp'==L_tes)/length(L_tes);      
    
        disp(['mdms dataset ', num2str(dataset_ind), ', sub ', num2str(sub_ind)])
        disp(['Naive acc (LDA) = ', num2str(acc_lda(sub_ind))]);
    end

    %%%%%%%%%%%%%%%%
    % save results %
    %%%%%%%%%%%%%%%%
    cd(save_dir);
    filename = ['results_lda_mdms_ds', num2str(dataset_ind)];
    save(filename,'acc_lda', 'senator_index_lib','weight_lib','pred_lda');
    cd(code_dir);
end