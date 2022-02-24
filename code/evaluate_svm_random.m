function evaluate_svm_random(config)

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
    filename = ['best_parameters_svm_acc_ds', num2str(dataset_ind), '.mat'];
    filename_lda = ['results_lda_random_ds', num2str(dataset_ind)]; % to use the same senator
    cd(data_dir);
    load(['F_c.mat']);
    load(filename);
    cd(save_dir);
    load(filename_lda);
    cd(code_dir);
    feat_dim = size(F{1,1,1},2);
    C = best_C; kernel_para = best_kernel_para;
    
    %%%%%%%%%%
    % buffer %
    %%%%%%%%%%
    acc_svm = zeros(1, sub_num);
    pred_svm = cell(1, sub_num);
    local_z_mu = zeros(sub_num, feat_dim);
    local_z_sigma = zeros(sub_num, feat_dim);

    %%%%%%%%%%%%%%%%%%%%%%%%%
    % train individual SVMs %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    SVMs = [];
    
    for sub_ind = 1:sub_num
        % preparation
        data = []; label = [];
        for trial_ind = 1:trial_num
            for mov_ind = 1:mov_num
                data = [data; F{sub_ind, trial_ind, mov_ind}];
                label = [label; c{sub_ind, trial_ind, mov_ind}];
            end
        end
        
        cmd = ['-q -s 0 -t 2 -b 1 -c ', num2str(C), '-g ', num2str(kernel_para)]; % RBF kernel (with probability estimates)
    
        % normalization
        [ZF, local_z_mu(sub_ind,:), local_z_sigma(sub_ind,:)] = zscore(data);
        SVMs = [SVMs; svmtrain(label, ZF, cmd)];
    end
    
    disp(['svm random dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' SVMs done'])
    
    %%%%%%%%%%%%%%
    % evaluation %
    %%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
    
        % preparation
        sub_ind_seq = 1:1:sub_num;
        sub_ind_seq(sub_ind) = [];       
        S_tes = []; L_tes = [];      
        
        if dataset_ind == 1
            for trial_ind = 3:5
               for mov_ind = 1:mov_num
                    S_tes = [S_tes; F{sub_ind, trial_ind, mov_ind}]; % 3rd to 5th trial
                    L_tes = [L_tes; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        else        
            for trial_ind = 5:6
                for mov_ind = 1:mov_num
                    S_tes = [S_tes; F{sub_ind, trial_ind, mov_ind}]; % 5th and 6th trials
                    L_tes = [L_tes; c{sub_ind, trial_ind, mov_ind}];
                end
            end 
        end
        
        % use selected sources (we already know which data is similar to the target data by optimization process)
        naive_prob = zeros(length(L_tes), mov_num);
        senator_index = senator_index_lib{sub_ind};
        
        for nb_ind = 1:nb_senator
            % nomalization
            S_tes_dammy = (S_tes - local_z_mu(sub_ind_seq(senator_index(nb_ind)), :)) ./ local_z_sigma(sub_ind_seq(senator_index(nb_ind)), :);
                    
            % naive performance
            [~, ~, temp_prob] = svmpredict(L_tes, S_tes_dammy, SVMs(sub_ind_seq(senator_index(nb_ind))), '-q -b 1');
            naive_prob = naive_prob + temp_prob;
        end
        
        % recognition
        [~, pred_naive_temp] = max(naive_prob');
        pred_svm{sub_ind} = pred_naive_temp;
        acc_svm(sub_ind) = sum(pred_naive_temp'==L_tes)/length(L_tes);
        
        disp(['random dataset ', num2str(dataset_ind), ', sub ', num2str(sub_ind)])
        disp(['Naive acc (SVM) = ', num2str(acc_svm(sub_ind))]);
    end

    %%%%%%%%%%%%%%%%
    % save results %
    %%%%%%%%%%%%%%%%
    cd(save_dir);
    filename = ['results_svm_random_ds', num2str(dataset_ind)];
    save(filename,'acc_svm', 'senator_index_lib',...
                  'pred_svm', 'best_C', 'best_kernel_para');
    cd(code_dir);
end