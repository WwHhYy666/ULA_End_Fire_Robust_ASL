% ========================================================================
% MAIN DOA (Direction of Arrival) ESTIMATION SCRIPT
% ========================================================================
% This script performs Direction of Arrival (DOA) estimation for audio signals
% using four methods: MVDR, SRP-PHAT, W-SRP-PHAT, and GCC-WLS.
% The script processes multiple audio files from a specified directory and 
% estimates the true angle of each audio file.
% ========================================================================

clear; clc; close all;
addpath("func");

%% ===================== Path Config =====================
audio_dir = "C:\Users\25401\Desktop\End-Fire\wav\wav_split_1m";
out_dir   = "C:\Users\25401\Desktop\End-Fire\result";
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% ===================== Metric Config =====================
th_deg = 6;          % Accuracy threshold (deg), output as ratio in [0,1]

%% ===================== Array / Signal Config =====================
fs    = 16000;
delta = 0.035;       % mic spacing (m)
M = 4;

rmic_raw = [ 0,        0, 0;
            -delta,    0, 0;
            -2*delta,  0, 0;
            -3*delta,  0, 0];

rmic = rmic_raw - mean(rmic_raw, 1);  % centered coordinates (for code1 methods)
max_dur_sec = 3.0;   % process first N seconds (set Inf to use full length)

%% =======================================================
%% ========== (A) Code1 Config: W-SRP-PHAT + GCC-WLS ======
%% =======================================================
T_celsius = 25;
RH_percent = 50;
P_kPa = 101.325;
c0 = speed_of_sound(T_celsius, RH_percent, P_kPa);

c = c0;
c_gcc = c0;

% ---- STFT Config (code1) ----
win_size      = 1024;
overlap_ratio = 0.75;
hopSize       = win_size - floor(win_size * overlap_ratio);

kwin    = kaiser(win_size, 1.9*pi);
kwsigma = sqrt(sum(kwin.^2) / hopSize);
window  = kwin(:) / kwsigma;

nfbands = win_size/2 + 1;
f_vect  = (0:win_size/2) * fs / win_size;

% Frequency range
fmin = 800;
fmax = 4500;
start_f = find(f_vect >= fmin, 1, 'first');
end_f   = find(f_vect <= fmax, 1, 'last');

% Frame Selection (Energy VAD)
keep_ratio = 0.15;

% Search Grid (code1)
phi_step = 0.2;
phiV = 0:phi_step:180;

% Weighting (code1)
freq_alpha = 2;
coh_alpha  = 2;

% GCC-WLS (code1)
gcc_oversample = 16;
gcc_beta = 0.8;

%% =======================================================
%% ========== (B) Code2 Config: MVDR/SRP-PHAT ONLY ========
c2 = c0;
rmic2 = rmic_raw;       

% ---- STFT Config (code2) ----
win_size2      = win_size;
overlap_ratio2 = overlap_ratio;
hopSize2       = hopSize;

window2  = window;

nfbands2 = nfbands;
f_vect2  = f_vect;

start_f2 = start_f;
end_f2   = end_f;

% ---- Scan grid & direction vectors (code2) ----
phiV2    = phiV;
phi_len2 = numel(phiV2);
theta_s  = 90;

V2 = [-sind(theta_s).*cosd(phiV2);
       sind(theta_s).*sind(phiV2);
       cosd(theta_s).*ones(1, phi_len2)];

% ---- SRP-PHAT mic pairs & tau precompute (code2) ----
mic_pairs2  = nchoosek(1:M, 2);
num_pairs2  = size(mic_pairs2, 1);

pair_vecs2  = zeros(num_pairs2, 3);
for k = 1:num_pairs2
    pair_vecs2(k,:) = rmic2(mic_pairs2(k,2),:) - rmic2(mic_pairs2(k,1),:);
end
tau_pairs2 = (pair_vecs2 * V2) / c2;

%% ===================== Batch Read =====================
files = dir(fullfile(audio_dir, "*.wav"));
if isempty(files)
    error("No .wav files found in: %s", audio_dir);
end

Nfiles = numel(files);
file_name  = strings(Nfiles,1);
true_angle = nan(Nfiles,1);

% ---- code2 methods (baseline): ONLY MVDR + SRP-PHAT ----
est_mvdr2  = nan(Nfiles,1);
est_srp2   = nan(Nfiles,1);

% ---- code1 methods (proposed) ----
est_wsrp   = nan(Nfiles,1);
est_gccwls = nan(Nfiles,1);

fprintf("========== Batch DOA (Far-field) ==========\n");
fprintf("Folder: %s\n", audio_dir);
fprintf("Files : %d\n", Nfiles);
fprintf("Methods: MVDR, SRP-PHAT, W-SRP-PHAT, GCC-WLS\n");
fprintf("===========================================\n\n");

for i = 1:Nfiles
    fpath = fullfile(files(i).folder, files(i).name);
    file_name(i)  = string(files(i).name);
    true_angle(i) = parse_angle_from_filename(files(i).name);

    fprintf("[%d/%d] %s | true=%.1f\n", i, Nfiles, files(i).name, true_angle(i));

    % ---- Read audio ----
    if ~exist(fpath, 'file')
        warning("File not found: %s", fpath);
        continue;
    end

    [Xall, file_fs] = audioread(fpath);
    if file_fs ~= fs
        Xall = resample(Xall, fs, file_fs);
    end

    if size(Xall,2) < M
        warning("Skip: need >=%d channels, got %d", M, size(Xall,2));
        continue;
    end
    Xall = Xall(:, 1:M);

    if isfinite(max_dur_sec)
        L = min(size(Xall,1), round(max_dur_sec * fs));
        Xall = Xall(1:L, :);
    end

    % ==========================================================
    % (1) Code2 baseline methods: MVDR / SRP-PHAT (NO MUSIC)
    [est_mvdr2(i), est_srp2(i)] = doa_two_methods_code2( ...
        Xall, fs, c2, rmic2, V2, phiV2, f_vect2, start_f2, end_f2, ...
        win_size2, hopSize2, nfbands2, window2, mic_pairs2, tau_pairs2, keep_ratio);

    % ==========================================================
    % (2) Code1 proposed methods: W-SRP-PHAT + GCC-WLS
    % ==========================================================
    Yf = stft_1(Xall, win_size, hopSize, nfbands, window);  % [nfrms x M x nfbands]
    nfrms = size(Yf,1);

    % ---- Frame selection (energy gate) ----
    ref_ch = 1;
    band_energy = squeeze(sum(abs(Yf(:, ref_ch, start_f:end_f)).^2, 3));  % [nfrms x 1]
    if all(band_energy == 0) || numel(band_energy) < 5
        frm_idx = 1:nfrms;
    else
        thr = quantile(band_energy, 1-keep_ratio);
        frm_idx = find(band_energy >= thr);
        if numel(frm_idx) < 10
            frm_idx = 1:nfrms; % fallback if too few frames
        end
    end

    Yf_use = Yf(frm_idx, :, :);

    est_wsrp(i)   = doa_srp_phat_wideband_farfield(Yf_use, phiV, rmic, c, f_vect, start_f, end_f, freq_alpha, coh_alpha);
    est_gccwls(i) = doa_gcc_wls_farfield(Yf_use, rmic, c_gcc, fs, win_size, f_vect, start_f, end_f, gcc_oversample, gcc_beta, freq_alpha, coh_alpha);

    fprintf("    MVDR=%.2f | SRP-PHAT=%.2f | W-SRP-PHAT=%.2f | GCC-WLS=%.2f\n\n", ...
        est_mvdr2(i), est_srp2(i), est_wsrp(i), est_gccwls(i));
end

%% ===================== Save Per-File Results =====================
T = table(file_name, true_angle, est_mvdr2, est_srp2, est_wsrp, est_gccwls, ...
    'VariableNames', {'Filename','TrueAngle','Est_MVDR','Est_SRP_PHAT','Est_W_SRP_PHAT','Est_GCCWLS'});

T.abs_err_mvdr   = abs(T.Est_MVDR      - T.TrueAngle);
T.abs_err_srp    = abs(T.Est_SRP_PHAT  - T.TrueAngle);
T.abs_err_wsrp   = abs(T.Est_W_SRP_PHAT- T.TrueAngle);
T.abs_err_gccwls = abs(T.Est_GCCWLS    - T.TrueAngle);

out_csv_file = fullfile(out_dir, "DOA_results_per_file.csv");
writetable(T, out_csv_file);
fprintf("Saved per-file results: %s\n", out_csv_file);

%% ===================== Metrics by Angle (MAE/RMSE/ACC) =====================
valid_idx = ~isnan(true_angle);
angles = unique(true_angle(valid_idx));
angles = sort(angles);

methods = ["MVDR","SRP-PHAT","W-SRP-PHAT","GCC-WLS"];
nA = numel(angles);
nM = numel(methods);

MAE  = nan(nA, nM);
RMSE = nan(nA, nM);
ACC  = nan(nA, nM);  % ratio in [0,1]

for ai = 1:nA
    ang = angles(ai);
    idx = (true_angle == ang);

    est_mat = [est_mvdr2(idx), est_srp2(idx), est_wsrp(idx), est_gccwls(idx)];
    tru_vec = true_angle(idx);

    for mi = 1:nM
        e = est_mat(:,mi) - tru_vec;
        e = e(~isnan(e));
        if isempty(e), continue; end

        MAE(ai,mi)  = mean(abs(e));
        RMSE(ai,mi) = sqrt(mean(e.^2));
        ACC(ai,mi)  = mean(abs(e) < th_deg);
    end
end

Tout = table(angles, ...
    MAE(:,1), RMSE(:,1), ACC(:,1), ...
    MAE(:,2), RMSE(:,2), ACC(:,2), ...
    MAE(:,3), RMSE(:,3), ACC(:,3), ...
    MAE(:,4), RMSE(:,4), ACC(:,4), ...
    'VariableNames', { ...
    'AngleDeg', ...
    'MAE_MVDR','RMSE_MVDR','ACC_MVDR', ...
    'MAE_SRP','RMSE_SRP','ACC_SRP', ...
    'MAE_WSRP','RMSE_WSRP','ACC_WSRP', ...
    'MAE_GCCWLS','RMSE_GCCWLS','ACC_GCCWLS'});

out_csv_metrics = fullfile(out_dir, "DOA_metrics_by_angle.csv");
writetable(Tout, out_csv_metrics);
fprintf("Saved metrics-by-angle: %s\n", out_csv_metrics);

fprintf("Done. CSV saved to: %s\n", out_dir);

%% ====================== Local Functions ======================

function ang = parse_angle_from_filename(fname)
% PARSE_ANGLE_FROM_FILENAME: Extract angle value from audio filename
% Attempts to extract angle from filename patterns like "20d..." or "100d..."
% Returns NaN if parsing fails or angle is outside valid range [0,180]
% 
% Input:  fname - Filename string
% Output: ang   - Angle in degrees, or NaN if not found
    ang = nan;
    s = string(fname);

    tok = regexp(s, '(\d{1,3})[dD]', 'tokens', 'once');
    if ~isempty(tok)
        ang = str2double(tok{1});
        if ang>=0 && ang<=180, return; end
    end

    tok2 = regexp(s, '(\d{1,3})', 'tokens');
    if ~isempty(tok2)
        ang = str2double(tok2{1}{1});
        if ~(ang>=0 && ang<=180), ang = nan; end
    end
end

function tau12 = tdoa_farfield(phi_deg_vec, m1, m2, rmic, c)
% TDOA_FARFIELD: Calculate Time Difference Of Arrival for far-field source
% Computes TDOA between microphone pairs assuming far-field (plane wave) propagation
% tau12 = tau_m1 - tau_m2 = -(U * d) / c
% Input: phi as scalar or vector
% Output: tau12 [N x 1] TDOA values
    phi = phi_deg_vec(:);
    U = [-cosd(phi), sind(phi), zeros(numel(phi),1)]; % [N x 3]
    d = rmic(m1,:) - rmic(m2,:);                      % [1 x 3]
    tau12 = -(U * d.') / c;                           % [N x 1]
end

function phi_hat = refine_peak_parabolic(phiV, P)
% REFINE_PEAK_PARABOLIC: Refine peak location using parabolic interpolation
% Performs parabolic interpolation around the discrete peak for sub-grid resolution
    [~, k] = max(P);
    if k <= 1 || k >= numel(P)
        phi_hat = phiV(k);
        return;
    end
    y1 = P(k-1); y2 = P(k); y3 = P(k+1);
    denom = (y1 - 2*y2 + y3);
    if abs(denom) < 1e-12
        phi_hat = phiV(k);
        return;
    end
    delta = 0.5*(y1 - y3) / denom;  % in grid steps
    step = phiV(2) - phiV(1);
    phi_hat = phiV(k) + delta*step;
    phi_hat = max(min(phi_hat, max(phiV)), min(phiV));
end

function doa = doa_srp_phat_wideband_farfield(Yf_use, phiV, rmic, c, f_vect, start_f, end_f, plot_alpha, coh_alpha)
% DOA_SRP_PHAT_WIDEBAND_FARFIELD: Weighted SRP-PHAT for wideband DOA (CODE1 proposed)
% Features: All pair processing, frequency-dependent weighting, coherence weighting
    M = size(rmic,1);
    phi_len = numel(phiV);
    P = zeros(phi_len,1);

    mic_pairs = nchoosek(1:M,2);
    num_pairs = size(mic_pairs,1);

    pair_len = zeros(num_pairs,1);
    for k = 1:num_pairs
        m1 = mic_pairs(k,1); m2 = mic_pairs(k,2);
        pair_len(k) = norm(rmic(m1,:) - rmic(m2,:));
    end
    pair_w = (pair_len / max(pair_len)).^2;  % endfire prefers longer baselines

    fmax = f_vect(end_f);

    for nbin = start_f:end_f
        f = f_vect(nbin);
        if f <= 0, continue; end
        wf = (f/fmax)^plot_alpha;

        for p = 1:num_pairs
            m1 = mic_pairs(p,1);
            m2 = mic_pairs(p,2);

            X1 = Yf_use(:, m1, nbin);
            X2 = Yf_use(:, m2, nbin);

            P12_raw = mean(X1 .* conj(X2), 1);
            P11 = mean(abs(X1).^2, 1);
            P22 = mean(abs(X2).^2, 1);

            coh = abs(P12_raw)^2 / (P11*P22 + 1e-12);
            coh = min(max(coh, 0), 1);
            if coh < 0.25
                continue; 
            end

            P12 = P12_raw / (abs(P12_raw) + 1e-12); % PHAT
            w = pair_w(p) * wf * (coh^coh_alpha);

            tau_vec = tdoa_farfield(phiV, m1, m2, rmic, c); % [phi_len x 1]
            P = P + w * real(P12 * exp(1j * 2*pi * f * tau_vec));
        end
    end

    P = abs(P);
    doa = refine_peak_parabolic(phiV, P);
end

function doa = doa_gcc_wls_farfield(Yf_use, rmic, c, fs, win_size, f_vect, start_f, end_f, os, beta, freq_alpha, coh_alpha)
% DOA_GCC_WLS_FARFIELD: GCC-PHAT TDOA with WLS fusion (CODE1 proposed)
% Per-pair TDOA extraction via oversampled GCC-PHAT, then WLS fusion with IRLS refinement
    M = size(rmic,1);

    mic_pairs = nchoosek(1:M,2);
    num_pairs = size(mic_pairs,1);

    pair_len = zeros(num_pairs,1);
    for k = 1:num_pairs
        m1 = mic_pairs(k,1); m2 = mic_pairs(k,2);
        pair_len(k) = norm(rmic(m1,:) - rmic(m2,:));
    end
    pair_w0 = (pair_len / max(pair_len)).^2;

    fmax = f_vect(end_f);

    cos_list = nan(num_pairs,1);
    w_list   = zeros(num_pairs,1);

    for p = 1:num_pairs
        m1 = mic_pairs(p,1);
        m2 = mic_pairs(p,2);

        if pair_len(p) < 0.05 
            continue; 
        end

        P12_pos = zeros(win_size/2+1, 1);
        coh_acc = 0;
        coh_cnt = 0;

        for nbin = start_f:end_f
            f = f_vect(nbin);
            if f <= 0, continue; end
            wf = (f/fmax)^freq_alpha;

            X1 = Yf_use(:, m1, nbin);
            X2 = Yf_use(:, m2, nbin);

            P12_raw = mean(X1 .* conj(X2), 1);
            P11 = mean(abs(X1).^2, 1);
            P22 = mean(abs(X2).^2, 1);

            coh = abs(P12_raw)^2 / (P11*P22 + 1e-12);
            coh = min(max(coh, 0), 1);

            % PHAT-beta
            P12_phat = P12_raw / ((abs(P12_raw) + 1e-12)^beta);

            P12_pos(nbin) = wf * (coh^coh_alpha) * P12_phat;

            coh_acc = coh_acc + coh;
            coh_cnt = coh_cnt + 1;
        end

        if coh_cnt == 0
            continue;
        end
        coh_avg = coh_acc / coh_cnt;

        P12_full = [P12_pos; conj(P12_pos(end-1:-1:2))]; % length = win_size

        N  = win_size;
        N2 = N * os;

        Xs = fftshift(P12_full);
        pad = N2 - N;
        Xs2 = [zeros(floor(pad/2),1); Xs; zeros(ceil(pad/2),1)];
        X2 = ifftshift(Xs2);

        r = real(ifft(X2));
        r = fftshift(r);

        lags = (-N2/2 : N2/2 - 1).' / (fs * os);

        tau_max = pair_len(p) / c;
        mask = (lags >= -1.05*tau_max) & (lags <= 1.05*tau_max);
        if ~any(mask), continue; end

        rr = r(mask);
        ll = lags(mask);

        [~, im] = max(abs(rr));
        tau_hat = ll(im);

        if im > 1 && im < numel(rr)
            y1 = abs(rr(im-1)); y2 = abs(rr(im)); y3 = abs(rr(im+1));
            denom = (y1 - 2*y2 + y3);
            if abs(denom) > 1e-12
                d = 0.5*(y1 - y3) / denom;
                dt = 1/(fs*os);
                tau_hat = tau_hat + d*dt;
            end
        end

        dx = rmic(m1,1) - rmic(m2,1);
        if abs(dx) < 1e-12
            continue;
        end
        cos_phi = (c * tau_hat) / dx;
        cos_phi = max(min(cos_phi, 1), -1);

        cos_list(p) = cos_phi;

        w_list(p) = pair_w0(p) * (coh_avg^coh_alpha);
    end

    valid = ~isnan(cos_list) & (w_list > 0);
    if ~any(valid)
        doa = nan;
        return;
    end

    cv = cos_list(valid);
    wv = w_list(valid);
    u_hat = sum(wv .* cv) / sum(wv);  
    
    for iter = 1:3  
        res = abs(cv - u_hat);
        wv_irls = wv .* exp(-5 * res); % lambda = 5
        if sum(wv_irls) < 1e-9, break; end
        u_hat = sum(wv_irls .* cv) / sum(wv_irls); 
    end
    cos_hat = u_hat;
    % ===============================
    
    cos_hat = max(min(cos_hat, 1), -1);
    doa = acosd(cos_hat);
end

% ==========================================================
% ============ CODE2 BASELINE METHODS (MVDR + SRP-PHAT) ==
% ==========================================================
function [est_mvdr, est_srp] = doa_two_methods_code2( ...
    Xall, ~, c, rmic, V, phiV, f_vect, start_f, end_f, ...
    win_size, hopSize, nfbands, window, mic_pairs, tau_pairs, keep_ratio)
% DOA_TWO_METHODS_CODE2: Baseline MVDR and SRP-PHAT methods (CODE2 reference)
% Compares two baseline far-field DOA methods: SRP-MVDR and SRP-PHAT

    M = size(rmic,1);
    phi_len = numel(phiV);
    num_pairs = size(mic_pairs,1);

    % STFT: [nfrms x M x nfbands]
    Yf_noisy = stft_1(Xall, win_size, hopSize, nfbands, window);
    nfrms = size(Yf_noisy, 1);

    % Frame selection (same rho as other methods)
    ref_ch = 1;
    band_energy = squeeze(sum(abs(Yf_noisy(:, ref_ch, start_f:end_f)).^2, 3));
    if all(band_energy == 0) || numel(band_energy) < 5
        frm_idx = 1:nfrms;
    else
        thr = quantile(band_energy, 1-keep_ratio);
        frm_idx = find(band_energy >= thr);
        if numel(frm_idx) < 10
            frm_idx = 1:nfrms;
        end
    end

    Yf_noisy_use = Yf_noisy(frm_idx, :, :);

    Pmvdr  = zeros(phi_len, 1);
    Psrp   = zeros(phi_len, 1);

    for nbin = start_f:end_f
        f = f_vect(nbin);

        Yf = Yf_noisy_use(:, 1:M, nbin);      % [nfrms x M]
        Yfm = Yf.';                        % [M x nfrms]

        Rxx = Yfm * conj(Yfm.');           % [M x M]
        inv_Rxx = inv(Rxx + 1e-6*eye(M));

        % steering matrix: A = exp(j*2*pi*f/c * (rmic*V))
        A = exp(1j * (2*pi*f/c) * (rmic * V));   % [M x phi_len]

        % ---- SRP-MVDR ----
        tmp = inv_Rxx * A;
        denom_mvdr = real(sum(conj(A) .* tmp, 1));
        Pmvdr = Pmvdr + 1 ./ (denom_mvdr(:) + eps);

        % ---- SRP-PHAT ----
        for k = 1:num_pairs
            mic1 = mic_pairs(k,1);
            mic2 = mic_pairs(k,2);

            X1 = Yf_noisy_use(:, mic1, nbin);
            X2 = Yf_noisy_use(:, mic2, nbin);

            P = X1 .* conj(X2);
            P = P ./ (abs(P) + 1e-10);

            Psum = sum(P);
            EXP = exp(1j * 2*pi*f * tau_pairs(k,:));
            Psrp = Psrp + real((Psum .* EXP).');
        end
    end

    [~, idx_mvdr]  = max(abs(Pmvdr));
    [~, idx_srp]   = max(abs(Psrp));

    est_mvdr  = phiV(idx_mvdr);
    est_srp   = phiV(idx_srp);
end

function c = speed_of_sound(T_celsius, RH_percent, P_kPa)
% SPEED_OF_SOUND: Calculate sound speed in moist air
% Uses Cramer-style approximation accounting for temperature, humidity, and pressure
    h = max(min(RH_percent, 100), 0) / 100;
    P = P_kPa * 10;

    e_s = 6.1121 * exp((18.678 - T_celsius/234.5) * (T_celsius/(257.14 + T_celsius)));
    e = h * e_s;

    x_w = e / P;
    c = 331.45 * sqrt(1 + (T_celsius / 273.15)) * (1 + 0.51 * x_w);
end
