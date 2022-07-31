cd ('/Users/nicole/OSU/GAN_TL/experiments/lyft_data')
% %% Clear
clc;
clear all;
%% close all opened files
close all;
%% set a random seed
seed=0;
rng(seed);

% normal_traces_x = csvread('lyft_good_traces_x.csv');
% normal_traces_y = csvread('lyft_good_traces_y.csv');
% 
% bad_traces_x = csvread('bad_traces_x.csv');
% bad_traces_y = csvread('bad_traces_y.csv');

normal_traces_x = csvread('../ecg/hr_traces_len_10/1_nsr_traces_len_10.csv');
normal_traces_y = csvread('../ecg/hr_traces_len_10_diff/1_nsr_traces_len_10.csv');

bad_traces_x = csvread('../ecg/hr_traces_len_10/4_afib_traces_len_10.csv');
bad_traces_y = csvread('../ecg/hr_traces_len_10_diff/4_afib_traces_len_10.csv');

%% plot traces
figure;
hold all;
	
% plot(0:1:49, awgn(normal_traces_y(1:20,:),25,'measured'),'g');
% plot(0:1:49, awgn(bad_traces_y(1:20,:),25,'measured'),'r');
plot(0:1:9, awgn(normal_traces_x(1:20,:),25,'measured'),'g');
plot(0:1:9, awgn(bad_traces_x(1:20,:),25,'measured'),'r');
% axis([0 100 10 50])
title('lyft traces');

%%
%applying enumerative solver 
cd ('/Users/nicole/OSU/GAN_TL/learningSTL/EnumerativeSolver')
traceTimeBegin = 0;
traceTimeHorizon = 49;   
timeRange = [traceTimeBegin, traceTimeHorizon];  
testRange = [-100, 100];
MCR_THRESH = 0.2;
s1 = struct('name', 'x', ...
            'ops', {{'<','>'}}, ...
            'params', {{'valx'}}, ...
            'timeRange', timeRange, ...
            'range', testRange ...
            );
s2 = struct('name', 'y', ...
            'ops', {{'<','>'}}, ...
            'params', {{'valy'}}, ...
            'timeRange', timeRange, ...
            'range', testRange ...
            );
signals = {s1, s2}; 
Traces0 = makeBreachTraceSystem(signals);%anomalous
Traces1 = makeBreachTraceSystem(signals);%normal 

Traces0_test = makeBreachTraceSystem(signals);%anomalous
Traces1_test = makeBreachTraceSystem(signals);%normal 


% options.numSignatureTraces = 3; 
options.numSignatureTraces = 5; 
numTraces = 10; 
yapp = YAP(signals, numTraces, options); 

%% train traces

%for jj = 1:8080
for jj = 1:50
    x=awgn(normal_traces_x(jj,:),25,'measured');
    y=awgn(normal_traces_y(jj,:),25,'measured');
%     t=0:49;
    t = 0:9;
    trace = [t' x' y'];
    Traces1.AddTrace(trace);
    yapp.addTrace(jj,trace);
    x=awgn(bad_traces_x(jj,:),25,'measured');
    y=awgn(bad_traces_y(jj,:),25,'measured');
    trace = [t' x' y'];
    Traces0.AddTrace(trace);
end
%% test traces
    
%for jj = 8081:10100
for jj = 51:100
    x=awgn(normal_traces_x(jj,:),25,'measured');
    y=awgn(normal_traces_y(jj,:),25,'measured');
%     t=0:49;
    t = 0:9;
    trace = [t' x', y'];
    Traces1_test.AddTrace(trace);
    yapp.addTrace(jj,trace);
    x=awgn(bad_traces_x(jj,:),25,'measured');
    y=awgn(bad_traces_y(jj,:),25,'measured');
    trace = [t' x', y'];
    Traces0_test.AddTrace(trace);
end
%% learning STL classifier with signature
clc
fprintf('running with optimization...\n')
j=1;
done=0;
f = formulaIterator(10, signals);
% f = formulaIterator(4, signals);
max_num_enumerating_formulas = 50;
best_mcr = 1;
best_formula = 0;
best_params = 0;
tic
while (1)  
% while (j < max_num_enumerating_formulas)
    formula = f.nextFormula();
    if length(fieldnames(get_params(formula))) > 4
        break
    end
    yapp.addTimeParams(formula);
    % check the equivalence of formulas
    if (yapp.isNew(formula))
        disp(formula)

        if length(fieldnames(get_params(formula))) == 1
            continue
        end
            
        % set parameter ranges
        params = fieldnames(get_params(formula));
        numparam=length(params);
        paramranges = zeros(numparam,2);
            
        for i=1:numparam
            if string(params(i,1)) == "valx"
                paramranges (i,:)=testRange;
            elseif string(params(i,1)) == "valy"
                paramranges (i,:)=testRange;
            elseif string(params(i,1)) == "tau_1"
%                 paramranges (i,:)=[0,100];
                paramranges (i,:)=[100,100];
            elseif string(params(i,1)) == "tau_2"
%                 paramranges (i,:)=[0,100];
                paramranges (i,:)=[100,100];
            elseif string(params(i,1)) == "tau_3"
%                 paramranges (i,:)=[0,100];
                paramranges (i,:)=[100,100];
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


%         if (all(monoDir1)>=0)
        if (all(monoDir1>=0))

            % obtain validity domain boundary
            mono=Monotonic_Bipartition (formula,paramranges,num_steps,uncertainty,Traces1,monoDir1);
            c1=reshape(mono.boundry_points,numparam,size(mono.boundry_points,2)/numparam)';

            % check points on validity domain boundary and choose
            % the point with MCR = 0
            for i = 1:size (c1,1)
                formula = set_params(formula,params, c1(i,:));
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
                    fprintf('\n\n');
                    fprintf('MCR  = %f\n',MCR);
                    fprintf('\n\n');

                    % check points on validity domain boundary and choose
                    % the point with MCR < 0.1
                    if MCR < best_mcr
                        best_mcr = MCR;
                        best_formula = formula;
                        best_params = params;
                        best_param_values = c1;
                        fprintf('Best MCR so far is %f\n', MCR);
                        fprintf('Best formula so far is\n');
                        fprintf(disp(formula));
                        fprintf('The values of best params so far are:\n');
                        for n = 1:size(params,1)
                            params(n)
                            c1(i,n)
                            fprintf('\n');
                        end
                    end

                    if MCR < MCR_THRESH
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
            if done==1
                break;
            end
        else
           fprintf('Not monotonic:\n') 
        end      
    else
        fprintf('Equivalent Formula:\n')
    end
    j=j+1;
end
 
% %%  learning STL classifier without signature
% % fprintf('***************************************************\n');
% % fprintf('running without optimization...\n')
% % j=1;
% % done=0;
% % f = formulaIterator(10, signals);
% % tic
% % while (1)
% %         
% %     formula = f.nextFormula();
% %     %set parameter ranges
% %     params = fieldnames(get_params(formula));
% %     numparam=length(params);
% %     paramranges = zeros(numparam,2);
% % 
% % 
% %     for i=1:numparam
% %         if string(params(i,1)) == "valx"
% %             paramranges (i,:)=[5, 50];
% %         elseif string(params(i,1)) == "tau_1"
% %             paramranges (i,:)=[0,100];
% %         elseif string(params(i,1)) == "tau_2"
% %             paramranges (i,:)=[0,100];
% %         elseif string(params(i,1)) == "tau_3"
% %             paramranges (i,:)=[0,100];
% %         end
% %     end
% %     % Monotonic_Bipartition function parameters
% %     uncertainty=10e-3;
% %     num_steps = 3; 
% %     n= numparam;
% %           
% %     % monotonicity direction for enumerated formulas, will
% %     % automate this part in next version
% %     switch j
% %         case 1 
% %             monoDir1=0;
% %         case 2
% %             monoDir1=1;
% %         case 3
% %             monoDir1=1;
% %         case 4
% %             monoDir1=0;
% %         case 5
% %             monoDir1=[0,0];
% %         case 6
% %             monoDir1=[1,0];
% %         case 7
% %             monoDir1=[0,1];
% %         case 8
% %             monoDir1=[1,1];
% %         case 9
% %             monoDir1=[1,0];
% %         case 10
% %             monoDir1=[0,0];
% %         case 11
% %             monoDir1=[0,1,0];
% %         case 12
% %             monoDir1=[1,1,0];
% %         case 13
% %             monoDir1=[1,1];
% %         case 14
% %             monoDir1=[0,1];
% %         case 15
% %             monoDir1=[0,0,1];
% %         case 16
% %             monoDir1=[1,0,1];
% %         otherwise
% %             monoDir1=-1;      
% %     end
% % 
% %     if (all(monoDir1)>=0)
% % 
% %         % obtain validity domain boundary
% %         mono=Monotonic_Bipartition (formula,paramranges,num_steps,uncertainty,Traces1,monoDir1);
% %         c1=reshape(mono.boundry_points,numparam,size(mono.boundry_points,2)/numparam)';
% % 
% %         % check points on validity domain boundary and choose
% %         % the point with MCR = 0
% %         for i = 1:size (c1,1)
% %             formula = set_params(formula,params, c1(i,:));
% %             robustness1(i,:)=Traces1.CheckSpec(formula);
% %             robustness2(i,:)=Traces0.CheckSpec(formula);
% % 
% %             if and(all(robustness1(i,:)> 0),all(robustness2(i,:)<0)) 
% %                 fprintf('\n\n');
% %                 fprintf('The learned STL formula is:\n');
% %                 fprintf('\n');
% %                 fprintf(disp(formula));
% %                 fprintf('\n\n');
% %                 fprintf('The values of parameters are:\n');
% %                 for n = 1:size(params,1)
% %                     params(n)
% %                     c1(i,n)
% %                     fprintf('\n');
% %                 end
% %                 fprintf('train MCR=0\n')
% %                 fprintf('Elapsed time without signature based optimization:\n')
% %                 toc
% %                 %plot the learned thresholds by our tool
% %                 y=[];
% %                 y(1,1:size(tspan,2))=c1(i,1);
% %                 plot(tspan,y,'b--','LineWidth',2);
% %                 xlabel('t(s)');
% %                 ylabel('v(m/s)');
% %                 done=1;
% %                 break; 
% %             else
% % 
% %                 TruePos=size(find (robustness1(i,:)> 0 == 1),2);
% %                 FalsePos=size(robustness1,2)-TruePos;
% % 
% %                 TrueNeg=size(find (robustness2(i,:) < 0 == 1),2);
% %                 FalseNeg=size(robustness2,2)-TrueNeg;
% % 
% %                 MCR = (FalsePos + FalseNeg)/(size(robustness1,2)+size(robustness2,2));
% % 
% %                 % check points on validity domain boundary and choose
% %                 % the point with MCR = 0.1
% %                 if MCR < 0.1
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
% %                     fprintf('train MCR = %f\n',MCR);
% %                     fprintf('Elapsed time without signature based optimization:\n')
% %                     toc
% %                     %plot the learned thresholds by our tool
% %                     y=[];
% %                     y(1,1:size(tspan,2))=c1(i,1);
% %                     plot(tspan,y,'b--','LineWidth',2);
% %                     xlabel('t(s)');
% %                     ylabel('v(m/s)');
% %                     done=1;
% %                     break;
% %                 end
% %             end
% % 
% %         end
% %         if done==1
% %             break;
% %         end
% %     end
% %    
% %     j=j+1;
% % end
%     
% %% computing testing MCR based on the learned STL classifier by our tool    
% pos = Traces1_test.CheckSpec('alw_[0,100] (x[t] < 35.8816)');
% neg = Traces0_test.CheckSpec('alw_[0,100] (x[t] < 35.8816)');
% mcr_test = (size(find (pos < 0),1) + size(find (neg > 0),1))/100;
% fprintf('test MCR = %f\n',mcr_test);

%% computing testing MCR based on the learned STL classifier by our tool    
% pos = Traces1_test.CheckSpec('x[t] > -1.3767');
% neg = Traces0_test.CheckSpec('x[t] > -1.3767');
% mcr_test = (size(find (pos < 0),2) + size(find (neg > 0),2))/100;
% fprintf('test MCR = %f\n',mcr_test);

pos = Traces1_test.CheckSpec('ev(x[t] > -0.9619)');
neg = Traces0_test.CheckSpec('ev(x[t] > -0.9619)');
mcr_test = (size(find (pos < 0),2) + size(find (neg > 0),2))/100;
fprintf('test MCR = %f\n',mcr_test);