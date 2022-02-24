function evaluate_lda_random(config)

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
    
    disp(['lda random dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' LDAs done'])

    %%%%%%%%%%%%%%
    % evaluation %
    %%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
        sub_ind_seq = 1:sub_num;
        sub_ind_seq(sub_ind) = [];
        
        % preparation
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
            
        % select senators
        temp_index = randperm(sub_num-1);
        senator_index = temp_index(1:nb_senator);
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
    
        disp(['random dataset ', num2str(dataset_ind), ', sub ', num2str(sub_ind)])
        disp(['Naive acc (LDA) = ', num2str(acc_lda(sub_ind))]);
    end

    %%%%%%%%%%%%%%%%
    % save results %
    %%%%%%%%%%%%%%%%
    cd(save_dir);
    filename = ['results_lda_random_ds', num2str(dataset_ind)];
    save(filename,'acc_lda','senator_index_lib','pred_lda');
    cd(code_dir);
end