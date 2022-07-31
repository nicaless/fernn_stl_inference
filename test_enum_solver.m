function test_enum_solver()
    %% Clear
    clc;
    clear all;
    %% close all opened files
    close all;
    %% set a random seed
    seed=0;
    rng(seed);
 
    %% Intitializations
%     tspan = 1:10;

    % READ IN DATA

%     good_traces = readmatrix('generated_hapt_data.csv');
%     good_traces = readmatrix('good_hapt_data.csv');
%      bad_traces = readmatrix('bad_hapt_data.csv');
%      good_traces = readmatrix('test_good_data.csv');
%      bad_traces = readmatrix('test_bad_data.csv');
     
%     good_traces = readmatrix('good_uav_dists.csv');
%     bad_traces = readmatrix('bad_uav_dists.csv');

    good_traces = readmatrix('good_pitch_data.csv');
    bad_traces = readmatrix('bad_pitch_data.csv');
     
    time_len = size(good_traces);
    time_len = time_len(2);
     
    tspan = 1:time_len;
    
    min_all = round(min(min(good_traces, [], 'all'), min(bad_traces, [], 'all')));
    max_all = round(max(max(good_traces, [], 'all'), max(bad_traces, [], 'all')));
    param_range = [min_all, max_all];
    
     MCR_thresh = 0.4;  % higher mcr_thresh during GAN_training
%     MCR_thresh = 0.1;

%     figure;
%     hold all;
%     	
%     plot(good_traces','g');
%     plot(bad_traces','r');

    %% APPLYING ENUMERATIVE SOLVER

    cd ('/Users/nicole/OSU/GAN_TL/learningSTL/EnumerativeSolver');
    traceTimeBegin = 1;
    traceTimeHorizon = time_len;   

    timeRange = [traceTimeBegin, traceTimeHorizon];  

    s1 = struct('name', 'x', ...
                'ops', {{'<','>'}}, ...
                'params', {{'valx'}}, ...
                'timeRange', timeRange, ...
                'range', param_range);     

    signals = {s1}; 
    Traces0 = makeBreachTraceSystem(signals);%anomalous
    Traces1 = makeBreachTraceSystem(signals);%normal 

    Traces0_test = makeBreachTraceSystem(signals);%anomalous
    Traces1_test = makeBreachTraceSystem(signals);%normal 

    % what are these parameters?
    options.numSignatureTraces = 3; 
    numTraces = 10;  
%     numTraces = 3;
    yapp = YAP(signals, numTraces, options); 


    %% train traces
    for jj = 1:50
        x=awgn(good_traces(jj,:),25,'measured');
        t=1:time_len;
        trace = [t' x'];
        Traces1.AddTrace(trace);
        yapp.addTrace(jj,trace);
        x=awgn(bad_traces(jj,:),25,'measured');
        trace = [t' x'];
        Traces0.AddTrace(trace);
    end
    %% test traces

    for jj = 51:100
        x=awgn(good_traces(jj,:),25,'measured');
        t=1:time_len;
        trace = [t' x'];
        Traces1_test.AddTrace(trace);
        x=awgn(bad_traces(jj,:),25,'measured');
        trace = [t' x'];
        Traces0_test.AddTrace(trace);
    end
    
%     figure;
%     hold all;
%     	
%     plot(good_traces(1:50,:)','g');
%     plot(bad_traces(1:50,:)','r');


    %% learning STL classifier with signature
    clc
    fprintf('running with optimization...\n')
    j=1;
    done=0;
%     f = formulaIterator(10, signals);
    f = formulaIterator(3, signals);  % restricting max length to 3
    tic
    while (1)  
        fprintf('getting next formula \n')
        fprintf('%d \n', j)
        formula = f.nextFormula();
        yapp.addTimeParams(formula);  
        % check the equivalence of formulas
        fprintf('check the equivalence of formulas \n')
        if (yapp.isNew(formula))
            fprintf('found new formula \n')
            % set parameter ranges
            params = fieldnames(get_params(formula));
            numparam=length(params);
            paramranges = zeros(numparam,2);

            for i=1:numparam
                if string(params(i,1)) == "valx"
                    paramranges (i,:)= param_range;
                elseif string(params(i,1)) == "tau_1"
                    paramranges (i,:)=[1,time_len];
                elseif string(params(i,1)) == "tau_2"
                    paramranges (i,:)=[1,time_len];
                elseif string(params(i,1)) == "tau_3"
                    paramranges (i,:)=[1,time_len];
                end
            end
            % Monotonic_Bipartition function parameters
            uncertainty=10e-3;
            num_steps = 3; 
            n= numparam;

            % monotonicity direction for enumerated formulas, will
            % automate this part in next version

            switch j
                case 1 
                    monoDir1=0;
                case 2
                    monoDir1=1;
                case 3
                    monoDir1=1;
                case 4
                    monoDir1=0;
                case 5
                    monoDir1=[0,0];
                case 6
                    monoDir1=[1,0];
                case 7
                    monoDir1=[0,1];
                case 8
                    monoDir1=[1,1];
                case 9
                    monoDir1=[1,0];
                case 10
                    monoDir1=[0,0];
                case 11
                    monoDir1=[0,1,0];
                case 12
                    monoDir1=[1,1,0];
                case 13
                    monoDir1=[1,1];
                case 14
                    monoDir1=[0,1];
                case 15
                    monoDir1=[0,0,1];
                case 16
                    monoDir1=[1,0,1];
                otherwise
                    monoDir1=-1;      
            end

            fprintf('begin calculating robustness \n')
            if (all(monoDir1)>=0)

                % obtain validity domain boundary
                fprintf('obtaining validity domain boundary \n')
                fprintf(disp(formula));
                mono=Monotonic_Bipartition (formula,paramranges,num_steps,uncertainty,Traces1,monoDir1);
                c1=reshape(mono.boundry_points,numparam,size(mono.boundry_points,2)/numparam)';

                % check points on validity domain boundary and choose
                % the point with MCR = 0
                fprintf('checking points on validity domain boundary \n')
                for i = 1:size (c1,1)
                    formula = set_params(formula,params, c1(i,:));
                    fprintf('calculating robustness values of point \n')
                    robustness1(i,:)=Traces1.CheckSpec(formula);
                    robustness2(i,:)=Traces0.CheckSpec(formula);

                    if and(all(robustness1(i,:)> 0),all(robustness2(i,:)<0)) 
                        fprintf('\n\n');
                        fprintf('The learned STL formula is:\n');
                        fprintf('\n');
                        fprintf(disp(formula));
                        fprintf('\n\n');
                        fprintf('The values of parameters are:\n');
                        for n = 1:size(params,1)
                            params(n)
                            c1(i,n)
                            fprintf('\n');
                        end
                        fprintf('train MCR=0\n')
                        fprintf('Elapsed time with signature based optimization:\n')
                        toc
                        done=1;
                        break; 
                    else

                        TruePos=size(find (robustness1(i,:)> 0 == 1),2);
                        FalsePos=size(robustness1,2)-TruePos;

                        TrueNeg=size(find (robustness2(i,:) < 0 == 1),2);
                        FalseNeg=size(robustness2,2)-TrueNeg;



                        MCR = (FalsePos + FalseNeg)/(size(robustness1,2)+size(robustness2,2));

                        % check points on validity domain boundary and choose
                        % the point with MCR < 0.1
                        if MCR < MCR_thresh
                            fprintf('\n\n');
                            fprintf('The learned STL formula is:\n');
                            fprintf('\n');
                            fprintf(disp(formula));
                            fprintf('\n\n');
                            fprintf('The values of parameters are:\n');
                            for n = 1:size(params,1)
                                params(n)
                                c1(i,n)
                                fprintf('\n');
                            end
                            fprintf('train MCR = %f\n',MCR);
                            fprintf('Elapsed time with signature based optimization:\n')
                            toc
                            done=1;
                            break;
                        end
                    end


                end
                if done==1 || j > 100
                    break;
                end
            end      
        end
        j=j+1;
    end


%     %  learning STL classifier without signature
% %     fprintf('***************************************************\n');
% %     fprintf('running without optimization...\n')
% %     j=1;
% %     done=0;
% %     %f = formulaIterator(10, signals);
% %     f = formulaIterator(2, signals);  % restricting max length of formulae
% %     tic
% %     while (1)
% %             
% %         formula = f.nextFormula();
% %         %set parameter ranges
% %         params = fieldnames(get_params(formula));
% %         numparam=length(params);
% %         paramranges = zeros(numparam,2);
% %     
% %     
% %         for i=1:numparam
% %             if string(params(i,1)) == "valx"
% %                 paramranges (i,:)= param_range;
% %             elseif string(params(i,1)) == "tau_1"
% %                 paramranges (i,:)=[1,10];
% %             elseif string(params(i,1)) == "tau_2"
% %                 paramranges (i,:)=[1,10];
% %             elseif string(params(i,1)) == "tau_3"
% %                 paramranges (i,:)=[1,10];
% %             end
% %         end
% %         % Monotonic_Bipartition function parameters
% %         uncertainty=10e-3;
% %         num_steps = 3; 
% %         n= numparam;
% %               
% %         % monotonicity direction for enumerated formulas, will
% %         % automate this part in next version
% %         switch j
% %             case 1 
% %                 monoDir1=0;
% %             case 2
% %                 monoDir1=1;
% %             case 3
% %                 monoDir1=1;
% %             case 4
% %                 monoDir1=0;
% %             case 5
% %                 monoDir1=[0,0];
% %             case 6
% %                 monoDir1=[1,0];
% %             case 7
% %                 monoDir1=[0,1];
% %             case 8
% %                 monoDir1=[1,1];
% %             case 9
% %                 monoDir1=[1,0];
% %             case 10
% %                 monoDir1=[0,0];
% %             case 11
% %                 monoDir1=[0,1,0];
% %             case 12
% %                 monoDir1=[1,1,0];
% %             case 13
% %                 monoDir1=[1,1];
% %             case 14
% %                 monoDir1=[0,1];
% %             case 15
% %                 monoDir1=[0,0,1];
% %             case 16
% %                 monoDir1=[1,0,1];
% %             otherwise
% %                 monoDir1=-1;      
% %         end
% %     
% %         if (all(monoDir1)>=0)
% %     
% %             % obtain validity domain boundary
% %             mono=Monotonic_Bipartition (formula,paramranges,num_steps,uncertainty,Traces1,monoDir1);
% %             c1=reshape(mono.boundry_points,numparam,size(mono.boundry_points,2)/numparam)';
% %     
% %             % check points on validity domain boundary and choose
% %             % the point with MCR = 0
% %             for i = 1:size (c1,1)
% %                 formula = set_params(formula,params, c1(i,:));
% %                 robustness1(i,:)=Traces1.CheckSpec(formula);
% %                 robustness2(i,:)=Traces0.CheckSpec(formula);
% %     
% %                 if and(all(robustness1(i,:)> 0),all(robustness2(i,:)<0)) 
% %                     fprintf('\n\n');
% %                     fprintf('The learned STL formula is:\n');
% %                     fprintf('\n');
% %                     fprintf(disp(formula));
% %                     fprintf('\n\n');
% %                     fprintf('The values of parameters are:\n');
% %                     for n = 1:size(params,1)
% %                         params(n)
% %                         c1(i,n)
% %                         fprintf('\n');
% %                     end
% %                     fprintf('train MCR=0\n')
% %                     fprintf('Elapsed time without signature based optimization:\n')
% %                     toc
% %                     %plot the learned thresholds by our tool
% %                     %y=[];
% %                     %y(1,1:size(tspan,2))=c1(i,1);
% %                     %plot(tspan,y,'b--','LineWidth',2);
% %                     %xlabel('t(s)');
% %                     %ylabel('v(m/s)');
% %                     done=1;
% %                     break; 
% %                 else
% %     
% %                     TruePos=size(find (robustness1(i,:)> 0 == 1),2);
% %                     FalsePos=size(robustness1,2)-TruePos;
% %     
% %                     TrueNeg=size(find (robustness2(i,:) < 0 == 1),2);
% %                     FalseNeg=size(robustness2,2)-TrueNeg;
% %     
% %                     MCR = (FalsePos + FalseNeg)/(size(robustness1,2)+size(robustness2,2));
% %     
% %                     % check points on validity domain boundary and choose
% %                     % the point with MCR = 0.1
% %                     if MCR < 0.1
% %                         fprintf('\n\n');
% %                         fprintf('The learned STL formula is:\n');
% %                         fprintf('\n');
% %                         fprintf(disp(formula));
% %                         fprintf('\n\n');
% %                         fprintf('The values of parameters are:\n');
% %                         for n = 1:size(params,1)
% %                             params(n)
% %                             c1(i,n)
% %                             fprintf('\n');
% %                         end
% %                         fprintf('train MCR = %f\n',MCR);
% %                         fprintf('Elapsed time without signature based optimization:\n')
% %                         toc
% %                         %plot the learned thresholds by our tool
% %                         %y=[];
% %                         %y(1,1:size(tspan,2))=c1(i,1);
% %                         %plot(tspan,y,'b--','LineWidth',2);
% %                         %xlabel('t(s)');
% %                         %ylabel('v(m/s)');
% %                         %done=1;
% %                         break;
% %                     end
% %                 end
% %     
% %             end
% %             if done==1
% %                 break;
% %             end
% %         end
% %        
% %         j=j+1;
% %     end
% 
    % Test MCR
    pos = Traces1_test.CheckSpec(formula);
    neg = Traces0_test.CheckSpec(formula);
    % mcr_test = (size(find (pos < 0),1) + size(find (neg > 0),1))/100;
    mcr_test = (length(find (pos < 0)) + length(find (neg > 0)))/100;
    fprintf('test MCR = %f\n',mcr_test);


    % TODO: append loss (mcr) to file

    % Save Formula to File
    fid = fopen('~/OSU/GAN_TL/formula.txt', 'wt');
    fprintf(fid, disp(formula));
    fclose(fid);

    fid = fopen('~/OSU/GAN_TL/formula_params.txt', 'wt');
    fprintf(fid, params{1});
    fprintf(fid, ',');
    fprintf(fid, '%d\n', round(c1(1), 2));
    fclose(fid);
