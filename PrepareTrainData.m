%-------------------------------------------------------------------------
%          Prepare dataset for Seq2Seq model training
%-------------------------------------------------------------------------

clear; close all; clc;
addpath('F:\idonashino\lab\SSA-for-Matlab-master\');

%% Main loop
% load dataset
% The dataset in this script can be downloaded from the following website:
% https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation
filepath = 'F:\idonashino\datasets\PPG_ABP\';
DATA = load([filepath, 'Part_1.mat']);
data = DATA.Part_1;

train_set = {};

for signal_num = 1: length(data)
    fprintf('%d / %d Loop...', signal_num, length(data));
    signal_group = data{1, signal_num};
    PPG_seq = signal_group(1, :);
    ABP_seq = signal_group(2, :);
    ECG_seq = signal_group(3, :);
    
    % decrease the samples in order to improve the speed
    PPG_seq = PPG_seq(1: round(end / 3));
    ABP_seq = ABP_seq(1: round(end / 3));
    
    try
        disp('Preprocessing...');
        % cycle segmentation
        PPG = segment_cycle(PPG_seq);
        ABP = segment_cycle(ABP_seq);
        
        % detrending and downsampling
        PPG = encode_signal(PPG);
        ABP = encode_signal(ABP);
                
        % figure;
        % subplot(2, 1, 1); plot(PPG);
        % subplot(2, 1, 2); plot(ABP);
        % pause;
        
        % remove abnormal situation
        sig_minus = abs(length(PPG) - length(ABP));
        if min(length(PPG), length(ABP)) < sig_minus
            continue;
        end
        
        train_set{end+1, 1} = PPG;
        train_set{end, 2} = ABP;
        
    catch ME
        fprintf('%d Loop is Error.', signal_num);
        disp('Error. Next loop is continued');
        continue;
    end
        
end

% Save as .mat and .txt
save('training_set.mat', 'train_set');
fid = fopen('training_set.txt', 'w');
for i = 1: size(train_set, 1)
    fprintf(fid, '%d ', train_set{i, 1}); fprintf(fid, '\t');
    fprintf(fid, '%d ', train_set{i, 2}); fprintf(fid, '\r\n');
end
fclose(fid);

%% Function tools

function [segmented_signal] = segment_cycle(signal, varargin)
%---------------------------------------------------------
%SEGMENT CYCLE, extract on cycle from a long PPG/ABP sequence
%Inputs
%   signal: PPG or ABP signal;
%   ssacom argument: default is 50;
%
%Outputs
%   segmented_signal: splited cycle;
% 
%Acknowledgement:
%   This script needs the function: ssacom, which can be found in the
%   following link: https://github.com/anton-a-tkachev/SSA-for-Matlab
%---------------------------------------------------------
    if nargin == 1
        ssa_num = 50;   % the number of decomposed signal by using SSA. Default.
    else
        ssa_num = varargin{1};
    end
        
    % SSA
    Q = ssacom(signal, ssa_num);
    
    % Find the onsets of each cycle
    S = sum(Q(:, 1: 2), 2);
    [~, loc_1] = findpeaks(-S);
    [~, loc_2] = findpeaks(S);
    
    seq_len = min(length(loc_1), length(loc_2));
    loc_1 = loc_1(1: seq_len); loc_2 = loc_2(1: seq_len);
    
    MinPeakDistance = mean(abs(loc_1 - loc_2));
    MinPeakProminence = mean(abs(signal(loc_1) - signal(loc_2)));
    
    [~, locs] = findpeaks(-signal, 'MinPeakDistance', MinPeakDistance, 'MinPeakProminence', MinPeakProminence);
    
    % Choose the forward onsets
    index = randi([1, length(locs)], 1, 1);
    segmented_signal = signal(locs(index): locs(index+1));
    
end

function [output] = encode_signal(signal)
%-----------------------------------------------------------------
%DETREND_SIGNAL, removing the trend, normalizing and mapping the signal
% to [0, 100] as integer.
%-----------------------------------------------------------------
    signal = mapminmax(signal, 0, 1);  % normalizing
    k = (signal(end) - signal(1)) / (length(signal) - 1);
    b = signal(1) - k;
    t = 1: length(signal);
    signal = signal - (k * t + b);      
    signal = round(100 * signal);    % mapping to [0, 100]
    output = signal(1: 2: end);   % downsampling   
end






