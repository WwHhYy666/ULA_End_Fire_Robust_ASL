function Yf = stft_1(x, win_size, hopSize, nfbands, window)
% STFT_1 Short-Time Fourier Transform for multi-channel audio (compatible with DOA estimation code)
% Inputs:
%   x        : Time-domain signal matrix [nSamples x nChannels]
%   win_size : STFT window length (in samples)
%   hopSize  : Hop size / frame shift (in samples)
%   nfbands  : Number of frequency bins (= win_size/2 + 1)
%   window   : Window function (column vector)
% Output:
%   Yf       : STFT result [nFrames x nChannels x nfbands] complex matrix

% Get signal dimensions
[nSamples, nChannels] = size(x);

% Compute total number of frames
nFrames = fix((nSamples - win_size) / hopSize) + 1;

% Initialize output matrix (nFrames x nChannels x nfbands)
Yf = zeros(nFrames, nChannels, nfbands);

% Perform STFT channel by channel
for ch = 1:nChannels
    % Extract the time-domain signal for the current channel
    x_ch = x(:, ch);

    % Process frame by frame
    for n = 1:nFrames
        % Compute start and end indices for the current frame
        start_idx = (n - 1) * hopSize + 1;
        end_idx = start_idx + win_size - 1;

        % Prevent out-of-bounds access (if last frame is shorter than win_size, zero-pad)
        if end_idx > nSamples
            frame = zeros(win_size, 1);
            frame(1:(nSamples - start_idx + 1)) = x_ch(start_idx:end);
        else
            frame = x_ch(start_idx:end_idx);
        end

        % Apply window
        frame_win = frame .* window;

        % FFT (keep positive-frequency part only)
        fft_frame = fft(frame_win);
        fft_frame = fft_frame(1:nfbands);  % Corresponds to frequencies from 0 to fs/2

        % Store into output array
        Yf(n, ch, :) = fft_frame;
    end
end

% Convert to complex matrix (ensure correct output type)
Yf = complex(Yf);
end
