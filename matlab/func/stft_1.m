function Yf = stft_1(x, win_size, hopSize, nfbands, window)
% STFT_1 多通道音频的短时傅里叶变换（适配DOA估计代码）
% 输入参数：
%   x        : 时域信号矩阵 [采样点数 x 通道数]
%   win_size : STFT窗长（样本数）
%   hopSize  : 帧移（样本数）
%   nfbands  : 频点数量（= win_size/2 + 1）
%   window   : 窗函数（列向量）
% 输出参数：
%   Yf       : STFT结果 [帧数 x 通道数 x 频点数] 复数矩阵

% 获取信号维度
[nSamples, nChannels] = size(x);
% 计算总帧数
nFrames = fix((nSamples - win_size) / hopSize) + 1;

% 初始化输出矩阵（帧数 x 通道数 x 频点数）
Yf = zeros(nFrames, nChannels, nfbands);

% 逐通道进行STFT
for ch = 1:nChannels
    % 提取当前通道的时域信号
    x_ch = x(:, ch);
    
    % 逐帧处理
    for n = 1:nFrames
        % 计算当前帧的起始位置
        start_idx = (n - 1) * hopSize + 1;
        end_idx = start_idx + win_size - 1;
        
        % 防止越界（最后一帧可能不足窗长，补0）
        if end_idx > nSamples
            frame = zeros(win_size, 1);
            frame(1:(nSamples - start_idx + 1)) = x_ch(start_idx:end);
        else
            frame = x_ch(start_idx:end_idx);
        end
        
        % 加窗
        frame_win = frame .* window;
        
        % 傅里叶变换（只取正频率部分）
        fft_frame = fft(frame_win);
        fft_frame = fft_frame(1:nfbands);  % 对应0到fs/2的频率
        
        % 存入输出矩阵
        Yf(n, ch, :) = fft_frame;
    end
end

% 转换为复数矩阵（确保输出格式正确）
Yf = complex(Yf);
end