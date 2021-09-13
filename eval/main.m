algorithms = {
     %'C2SNet';
     %'BMPM';
     %'R3Net';
     %'RAS';
     %'PiCANet_VGG';
     %'PiCANet_R';

     %'BANet_VGG';
     %'BANet_R';
     %'PAGE';
     %'TDBU';
     %'SCRN';
     %'BASNet';
     %'CPD_VGG';
     %'CPD_R';
     %'GateNet_VGG';
     %'GateNet_R';
     %'PoolNet_VGG';
     %'PoolNet_R';
     %'EGNet_VGG';
     %'EGNet_R';
     %'AFNet';
     %'MLMS';
     %'F3Net'
     %'ITSDNet_VGG'
     %'ITSDNet_R'
     %'U2Net'
     %'DFI'
     %'LDF'
     %'MINet'
     %'GCPANet'
     'CTDNet'
    };

datasets = {
              %'ECSSD';
              %'PASCAL-S';
              %'DUTS';
              %'HKU-IS';
              %'DUT-OMRON';
              %'SOD';
              %'THUR15K';
              '1_360-SSOD';
              '360-SOD-r512_256-te';
    };


for i = 1:numel(algorithms)
    alg = algorithms{i};
    fprintf('The method is %s\n', alg);
    for j = 1:numel(datasets)
        dataset      = datasets{j};
        predpath     = ['./maps/' alg '/' dataset '/'];
        maskpath     = ['../data/' dataset '/mask/'];
        if ~exist(predpath, 'dir'), continue; end

        names = dir(['./maps/' alg '/' dataset '/*.png']);
        names = {names.name}';
        wfm          = 0; mae    = 0; sm     = 0; fm     = 0; prec   = 0; rec    = 0; em     = 0;
        score1       = 0; score2 = 0; score3 = 0; score4 = 0; score5 = 0; score6 = 0; score7 = 0;

        results      = cell(numel(names), 6);
        ALLPRECISION = zeros(numel(names), 256);
        ALLRECALL    = zeros(numel(names), 256);
        file_num     = false(numel(names), 1);
        for k = 1:numel(names)
            name          = names{k,1};
            results{k, 1} = name;
            file_num(k)   = true;
            fgpath        = [predpath name];
            fg            = imread(fgpath);
            
            if strcmp(alg, 'PoolNet_R')
                gtpath = [maskpath strrep(name, '_sal_fuse.', '.')];
            elseif strcmp(alg, 'DFI')
                gtpath = [maskpath strrep(name, '_sal_fuse.', '.')];
            elseif strcmp(alg, 'C2SNet')
                gtpath = [maskpath strrep(name, '_c2s.', '.')];
           elseif strcmp(alg, 'BMPM')
                gtpath = [maskpath strrep(name, '.jpg', '.png')];
            elseif strcmp(alg, 'RAS')
                gtpath = [maskpath strrep(name, '_ras.', '.')];
            else
                gtpath = [maskpath name];
            end
            gt = imread(gtpath);

            if length(size(fg)) == 3, fg = fg(:,:,1); end
            if length(size(gt)) == 3, gt = gt(:,:,1); end
            fg = imresize(fg, size(gt)); 
            fg = mat2gray(fg); 
            gt = mat2gray(gt);
            if max(fg(:)) == 0 || max(gt(:)) == 0, continue; end
            
            gt(gt>=0.5) = 1; gt(gt<0.5) = 0; gt = logical(gt);
            score1                   = MAE(fg, gt);
            [score2, score3, score4] = Fmeasure(fg, gt, size(gt)); 
            score5                   = wFmeasure(fg, gt); 
            score6                   = Smeasure(fg, gt);
            score7                   = Emeasure(fg, gt);
            mae                      = mae  + score1;
            prec                     = prec + score2;
            rec                      = rec  + score3;
            fm                       = fm   + score4;
            wfm                      = wfm  + score5;
            sm                       = sm   + score6;
            em                       = em   + score7;
            results{k, 2}            = score1; 
            results{k, 3}            = score4; 
            results{k, 4}            = score5; 
            results{k, 5}            = score6;
            results{k, 6}            = score7;
            [precision, recall]      = PRCurve(fg*255, gt);
            ALLPRECISION(k, :)       = precision;
            ALLRECALL(k, :)          = recall;
        end
        prec     = mean(ALLPRECISION(file_num,:), 1);   
        rec      = mean(ALLRECALL(file_num,:), 1);
        maxF     = max(1.3*prec.*rec./(0.3*prec+rec+eps));
        file_num = double(file_num);
        fm       = fm  / sum(file_num);
        mae      = mae / sum(file_num); 
        wfm      = wfm / sum(file_num); 
        sm       = sm  / sum(file_num); 
        em       = em  / sum(file_num);
        fprintf('%s: %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f\n', dataset, maxF, fm, mae, wfm, sm, em)
        save_path = ['./result' filesep alg filesep dataset filesep];
        if ~exist(save_path, 'dir'), mkdir(save_path); end
        save([save_path 'results.mat'], 'results');
        save([save_path 'prec.mat'], 'prec');
        save([save_path 'rec.mat'], 'rec');
    end
end
