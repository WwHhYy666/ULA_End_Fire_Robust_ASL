clear; clc; close all;
addpath("func");

%% ===================== Path Config =====================
audio_dir = "D:\acoustic\asl\EXPERIMENT\wav_split_1m";
out_dir   = "D:\acoustic\asl\EXPERIMENT";
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
T_celsius = 25;       % ambient temperature (C)
RH_percent = 50;      % relative humidity (%), adjust if known
P_kPa = 101.325;      % air pressure (kPa)
c0 = speed_of_sound(T_celsius, RH_percent, P_kPa);

c = c0;               % speed of sound for W-SRP-PHAT
c_gcc = c0;           % speed of sound for GCC-WLS

% ---- STFT Config (code1) ----
win_size      = 1024;
overlap_ratio = 0.75;
hopSize       = win_size - floor(win_size * overlap_ratio);

kwin    = kaiser(win_size, 1.9*pi);
kwsigma = sqrt(sum(kwin.^2) / hopSize);
window  = kwin(:) / kwsigma;

nfbands = win_size/2 + 1;
f_vect  = (0:win_size/2) * fs / win_size;

% Frequency range (W-SRP-PHAT)
fmin = 800;
fmax = 4500;
start_f = find(f_vect >= fmin, 1, 'first');
end_f   = find(f_vect <= fmax, 1, 'last');

% Frame Selection (Energy VAD)
keep_ratio = 0.30;

% Search Grid (code1)
phi_step = 0.2;
phiV = 0:phi_step:180;

% Weighting (code1)
freq_alpha = 2;
coh_alpha  = 1;

% GCC-WLS (guided robust WLS from gcc_wls_experiment.m)
gcc_fmin = 1000;
gcc_fmax = 4200;
start_f_gcc = find(f_vect >= gcc_fmin, 1, 'first');
end_f_gcc   = find(f_vect <= gcc_fmax, 1, 'last');

keep_ratio_energy = 0.30;
keep_ratio_msc    = 0.60;

gcc_beta = 0.7;
gcc_freq_alpha = 3;
gcc_coh_alpha  = 1;
gcc_oversample = 16;

theta_grid = [0:0.10:25, 25.25:0.25:154.75, 155:0.10:180];
srp_q = 1.0;
tau_win_ratio = 0.18;
psr_delta = 1.0;
psr_guard_ratio = 0.08;
huber_k_u = 0.08;
irls_iter = 8;
guided_iters = 2;

%% =======================================================
%% ========== (B) Code2 Config: MVDR/SRP-PHAT ONLY ========
c2 = c0;               
rmic2 = rmic_raw;       

% ---- STFT Config (code2) ----
win_size2      = 512;
overlap_ratio2 = 0.5;
hopSize2       = win_size2 - fix(win_size2 * overlap_ratio2);

kwin2    = kaiser(win_size2, 1.9*pi);
kwsigma2 = sqrt(sum(kwin2.^2)/hopSize2);
window2  = kwin2(:)/kwsigma2;

nfbands2 = fix(win_size2/2) + 1;
f_vect2  = (0:fix(win_size2/2)) * fs / win_size2;

start_f2 = fix(1000 / (fs / win_size2));
end_f2   = fix(4000 / (fs / win_size2));

% ---- Scan grid & direction vectors (code2) ----
phiV2    = 0:1:180;
phi_len2 = numel(phiV2);
theta_s  = 90;

V2 = [-sind(theta_s).*cosd(phiV2);
       sind(theta_s).*sind(phiV2);
       cosd(theta_s).*ones(1, phi_len2)];

% ---- SRP-PHAT mic pairs & tau precompute (code2) ----
mic_pairs2  = [1, 2; 1, 3; 1, 4];
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
        win_size2, hopSize2, nfbands2, window2, mic_pairs2, tau_pairs2);

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

    est_wsrp(i) = doa_srp_phat_wideband_farfield(Yf_use, phiV, rmic, c, f_vect, start_f, end_f, freq_alpha, coh_alpha);

    Yf_gcc_full = stft_multi(Xall, win_size, hopSize, nfbands, window);
    frm_idx_gcc = select_frames_energy_msc(Yf_gcc_full, start_f_gcc, end_f_gcc, keep_ratio_energy, keep_ratio_msc);
    Yf_gcc = Yf_gcc_full(frm_idx_gcc, :, :);
    est_gccwls(i) = doa_gcc_wls_farfield_v3( ...
        Yf_gcc, rmic, c_gcc, fs, win_size, f_vect, start_f_gcc, end_f_gcc, ...
        gcc_oversample, gcc_beta, gcc_freq_alpha, gcc_coh_alpha, theta_grid, srp_q, ...
        tau_win_ratio, psr_delta, psr_guard_ratio, huber_k_u, irls_iter, guided_iters);

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
% Try to parse true angle from filename.
% Priority pattern: "20d..." -> angle=20
% Fallback: first integer found
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

function Yf = stft_multi(x, win_size, hopSize, nfbands, window)
    [nSamples, nCh] = size(x);
    nFrames = fix((nSamples - win_size) / hopSize) + 1;
    Yf = zeros(nFrames, nCh, nfbands);
    for ch = 1:nCh
        xx = x(:,ch);
        for n = 1:nFrames
            st = (n-1)*hopSize + 1;
            ed = st + win_size - 1;
            frame = xx(st:ed) .* window;
            F = fft(frame);
            Yf(n,ch,:) = F(1:nfbands);
        end
    end
    Yf = complex(Yf);
end

function tau12 = tdoa_farfield(phi_deg_vec, m1, m2, rmic, c)
% Far-field TDOA: tau12 = tau_m1 - tau_m2
% Support phi as scalar or vector
    phi = phi_deg_vec(:);
    U = [-cosd(phi), sind(phi), zeros(numel(phi),1)]; % [N x 3]
    d = rmic(m1,:) - rmic(m2,:);                      % [1 x 3]
    tau12 = -(U * d.') / c;                           % [N x 1]
end

function phi_hat = refine_peak_parabolic(phiV, P)
% Parabolic peak refinement on discrete spectrum
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
% SRP-PHAT (far-field) using all pairs + baseline weighting + coherence weighting (CODE1)
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

            P12 = P12_raw / (abs(P12_raw) + 1e-12); % PHAT
            w = pair_w(p) * wf * (coh^coh_alpha);

            tau_vec = tdoa_farfield(phiV, m1, m2, rmic, c); % [phi_len x 1]
            P = P + w * real(P12 * exp(1j * 2*pi * f * tau_vec));
        end
    end

    P = abs(P);
    doa = refine_peak_parabolic(phiV, P);
end

function doa = doa_gcc_wls_farfield(Yuse, rmic, c, fs, win_size, f_vect, start_f, end_f, ...
    os, beta, freq_alpha, coh_alpha, theta_grid, srp_q, tau_win_ratio, psr_delta, psr_guard_ratio, ...
    huber_k_u, irls_iter, guided_iters)
% GCC-SRP guided robust WLS (far-field), adapted from gcc_wls_experiment.m
    [gcc_curves, lag_vec, dx_list, pair_w_geom, coh_avg_list] = ...
        build_gcc_curves(Yuse, rmic, c, fs, win_size, f_vect, start_f, end_f, ...
                         os, beta, freq_alpha, coh_alpha);

    theta0 = srp_search_theta(theta_grid, gcc_curves, lag_vec, dx_list, c, ...
                              pair_w_geom, coh_avg_list, srp_q);

    theta_hat = theta0;
    for it = 1:guided_iters
        u0 = cosd(theta_hat);

        [u_meas, w_meas] = guided_pick_and_weight( ...
            gcc_curves, lag_vec, dx_list, c, u0, ...
            pair_w_geom, coh_avg_list, tau_win_ratio, psr_delta, psr_guard_ratio);

        if isempty(u_meas)
            theta_hat = theta0;
            break;
        end

        u_hat = irls_scalar_huber(u_meas, w_meas, huber_k_u, irls_iter);
        theta_hat = acosd(max(min(u_hat,1),-1));
    end

    doa = theta_hat;
end

function frm_idx = select_frames_energy_msc(Yf, start_f, end_f, keepE, keepC)
    [nfrms, M, ~] = size(Yf);
    ref = 1;

    % energy gate
    bandE = squeeze(sum(abs(Yf(:,ref,start_f:end_f)).^2, 3));
    if all(bandE==0) || numel(bandE)<5
        frm_idx = 1:nfrms; return;
    end
    thrE = quantile(bandE, 1-keepE);
    idxE = find(bandE >= thrE);
    if numel(idxE) < 10, frm_idx = 1:nfrms; return; end

    % frame MSC gate (frequency-domain expectation for each frame)
    pairs = nchoosek(1:M,2);
    msc = zeros(numel(idxE),1);
    epsv = 1e-12;
    for ii = 1:numel(idxE)
        t = idxE(ii);
        s = 0;
        for p = 1:size(pairs,1)
            m1 = pairs(p,1); m2 = pairs(p,2);
            X1 = squeeze(Yf(t,m1,start_f:end_f)).';
            X2 = squeeze(Yf(t,m2,start_f:end_f)).';
            P12 = mean(X1 .* conj(X2));
            P11 = mean(abs(X1).^2);
            P22 = mean(abs(X2).^2);
            s = s + (abs(P12)^2 / (P11*P22 + epsv));
        end
        msc(ii) = s / size(pairs,1);
    end
    thrC = quantile(msc, 1-keepC);
    idxC = idxE(msc >= thrC);
    if numel(idxC) < 10
        frm_idx = idxE;
    else
        frm_idx = idxC;
    end
end

function [gcc_curves, lag_vec, dx_list, w_geom, coh_avg_list] = ...
    build_gcc_curves(Yuse, rmic, c, fs, win_size, f_vect, start_f, end_f, ...
                     os, beta, freq_alpha, coh_alpha)

    M = size(rmic,1);
    pairs = nchoosek(1:M,2);
    P = size(pairs,1);

    dx_list = zeros(P,1);
    d_list  = zeros(P,1);
    for p = 1:P
        m1=pairs(p,1); m2=pairs(p,2);
        dx_list(p) = rmic(m1,1) - rmic(m2,1);
        d_list(p)  = norm(rmic(m1,:) - rmic(m2,:));
    end
    w_geom = (d_list / max(d_list)).^2;

    N  = win_size;
    N2 = N*os;

    lag_vec = (-N2/2 : N2/2-1).' / (fs*os);
    gcc_curves = zeros(numel(lag_vec), P);
    coh_avg_list = zeros(P,1);

    fmax = f_vect(end_f);
    epsv = 1e-12;

    for p = 1:P
        m1=pairs(p,1); m2=pairs(p,2);

        P12_pos = zeros(N/2+1,1);
        coh_acc = 0; coh_cnt = 0;

        for k = start_f:end_f
            f = f_vect(k);
            wf = (f/fmax)^freq_alpha;

            X1 = Yuse(:,m1,k);
            X2 = Yuse(:,m2,k);

            P12 = mean(X1 .* conj(X2));
            P11 = mean(abs(X1).^2);
            P22 = mean(abs(X2).^2);

            coh = abs(P12)^2 / (P11*P22 + epsv);
            coh = min(max(coh,0),1);

            P12_phat = P12 / ((abs(P12)+epsv)^beta);
            P12_pos(k) = wf * (coh^coh_alpha) * P12_phat;

            coh_acc = coh_acc + coh;
            coh_cnt = coh_cnt + 1;
        end

        coh_avg_list(p) = coh_acc / max(coh_cnt,1);

        P12_full = [P12_pos; conj(P12_pos(end-1:-1:2))];

        Xs  = fftshift(P12_full);
        pad = N2 - N;
        Xs2 = [zeros(floor(pad/2),1); Xs; zeros(ceil(pad/2),1)];
        X2  = ifftshift(Xs2);

        r = real(ifft(X2));
        r = fftshift(r);

        % per-pair normalize (important: avoid one pair dominates SRP)
        r = r / (max(abs(r)) + epsv);

        gcc_curves(:,p) = r;
    end
end

function theta0 = srp_search_theta(theta_grid, gcc_curves, lag_vec, dx_list, c, w_geom, coh_avg, q)
    P = numel(dx_list);
    score = zeros(numel(theta_grid),1);
    epsv = 1e-12;

    for i = 1:numel(theta_grid)
        th = theta_grid(i);
        u  = cosd(th);
        s  = 0;

        for p = 1:P
            dx = dx_list(p);
            if abs(dx) < 1e-12, continue; end
            tau = dx*u/c;

            g = interp1(lag_vec, gcc_curves(:,p), tau, 'linear', 0);
            g = abs(g);

            wp = w_geom(p) * (coh_avg(p) + epsv); % 简单但有效
            s  = s + wp * (g^q);
        end
        score(i) = s;
    end

    [~, imax] = max(score);
    theta0 = theta_grid(imax);

    % parabolic refine around peak on theta-grid
    if imax>1 && imax<numel(theta_grid)
        y1 = score(imax-1); y2 = score(imax); y3 = score(imax+1);
        denom = (y1 - 2*y2 + y3);
        if abs(denom) > 1e-12
            d = 0.5*(y1 - y3)/denom;
            stepL = theta_grid(imax) - theta_grid(imax-1);
            stepR = theta_grid(imax+1) - theta_grid(imax);
            step  = min(stepL, stepR);
            theta0 = theta0 + d*step;
            theta0 = max(min(theta0,180),0);
        end
    end
end

function [u_meas, w_meas] = guided_pick_and_weight( ...
    gcc_curves, lag_vec, dx_list, c, u0, w_geom, coh_avg, tau_win_ratio, psr_delta, psr_guard_ratio)

    P = numel(dx_list);
    epsv = 1e-12;

    u_meas = [];
    w_meas = [];

    for p = 1:P
        dx = dx_list(p);
        if abs(dx) < 1e-12, continue; end

        tau_pred = dx*u0/c;

        tau_max = abs(dx)/c;
        tau_win = max( 2*(lag_vec(2)-lag_vec(1)), tau_win_ratio*tau_max );

        idx = find(lag_vec >= (tau_pred - tau_win) & lag_vec <= (tau_pred + tau_win));
        if numel(idx) < 5, continue; end

        r = gcc_curves(:,p);
        a = abs(r(idx));
        [~, im] = max(a);
        k0 = idx(im);

        % parabolic refine on abs(r)
        tau_hat = lag_vec(k0);
        if k0>1 && k0<numel(lag_vec)
            y1 = abs(r(k0-1)); y2 = abs(r(k0)); y3 = abs(r(k0+1));
            denom = (y1 - 2*y2 + y3);
            if abs(denom) > 1e-12
                d = 0.5*(y1 - y3)/denom;
                dt = lag_vec(2) - lag_vec(1);
                tau_hat = tau_hat + d*dt;
            end
        end

        u_p = (c * tau_hat) / dx;
        u_p = max(min(u_p,1),-1);

        % PSR around peak (exclude guard neighborhood)
        guard = psr_guard_ratio*tau_max;
        mask = (lag_vec < (tau_hat-guard)) | (lag_vec > (tau_hat+guard));
        side = median(abs(r(mask)));
        psr  = abs(interp1(lag_vec, abs(r), tau_hat, 'linear', abs(r(k0)))) / (side + epsv);
        psr  = min(max(psr,1),100);

        w_p = w_geom(p) * (coh_avg(p) + epsv) * (psr^psr_delta);

        u_meas(end+1,1) = u_p; %#ok<AGROW>
        w_meas(end+1,1) = w_p; %#ok<AGROW>
    end
end

function u = irls_scalar_huber(us, ws, k, maxIter)
    epsv = 1e-12;
    u = sum(ws.*us) / (sum(ws)+epsv);

    for it = 1:maxIter
        r = u - us;
        ar = abs(r);
        wrob = ones(size(r));
        idx = ar > k;
        wrob(idx) = k ./ (ar(idx) + epsv);

        weff = ws .* wrob;
        u_new = sum(weff.*us) / (sum(weff)+epsv);

        if abs(u_new-u) < 1e-6
            u = u_new; break;
        end
        u = u_new;
    end
    u = max(min(u,1),-1);
end
function [est_mvdr, est_srp] = doa_two_methods_code2( ...
    Xall, fs, c, rmic, V, phiV, f_vect, start_f, end_f, ...
    win_size, hopSize, nfbands, window, mic_pairs, tau_pairs)

    M = size(rmic,1);
    phi_len = numel(phiV);
    num_pairs = size(mic_pairs,1);

    % STFT: [nfrms x M x nfbands]
    Yf_noisy = stft_1(Xall, win_size, hopSize, nfbands, window);

    Pmvdr  = zeros(phi_len, 1);
    Psrp   = zeros(phi_len, 1);

    for nbin = start_f:end_f
        f = f_vect(nbin);

        % 取该频点所有帧的数据
        Yf = Yf_noisy(:, 1:M, nbin);      % [nfrms x M]
        Yfm = Yf.';                        % [M x nfrms]

        % 自相关矩阵
        Rxx = Yfm * conj(Yfm.');           % [M x M]
        inv_Rxx = inv(Rxx + 1e-6*eye(M));

        % steering matrix: A = exp(j*2*pi*f/c * (rmic*V))
        A = exp(1j * (2*pi*f/c) * (rmic * V));   % [M x phi_len]

        % ---- MVDR ----
        tmp = inv_Rxx * A;
        denom_mvdr = real(sum(conj(A) .* tmp, 1));
        Pmvdr = Pmvdr + 1 ./ (denom_mvdr(:) + eps);

        % ---- SRP-PHAT ----
        for k = 1:num_pairs
            mic1 = mic_pairs(k,1);
            mic2 = mic_pairs(k,2);

            X1 = Yf_noisy(:, mic1, nbin);
            X2 = Yf_noisy(:, mic2, nbin);

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


function doa = doa_gcc_wls_farfield_v3( ...
    Yf_use, rmic, c, fs, win_size, f_vect, start_f, end_f, ...
    os, beta, freq_alpha, coh_alpha, theta_grid, srp_q, ...
    tau_win_ratio, psr_delta, psr_guard_ratio, huber_k_u, irls_iter, guided_iters)
% GCC-WLS (stable+) — keep the good-performing pipeline from gcc_wls_experiment,
% then add ONLY stability improvements that should not distort angle trends:
%   A) SRP Top-L multi-start (small L) to avoid wrong init under multipath
%   B) Adaptive tau window: larger around broadside, tighter near end-fire
%   C) MAD outlier rejection before IRLS fusion (angle-neutral)
%
% NOTE: No end-fire-specific geometry damping and no post-refinement SRP here,
% to avoid "over-helping" certain angles and creating unnatural ACC patterns.

    % ---- Parameters for stability (local, GCC-WLS only) ----
    srpTopL      = 3;     % 3~5
    srpMinSepDeg = 5.0;   % ensure distinct hypotheses
    tauWinGain   = 0.8;   % broadside window expansion factor
    madBaseK     = 3.0;   % MAD threshold (2.5~3.5)
    madTighten   = 0.5;   % tighten near end-fire (0~1)
    madMinRemain = 2;     % keep at least 2 pairs

    % --- Build GCC curves for all pairs ---
    [gcc_curves, lag_vec, dx_list, w_geom, coh_avg_list] = ...
        build_gcc_curves(Yf_use, rmic, c, fs, win_size, f_vect, start_f, end_f, ...
                         os, beta, freq_alpha, coh_alpha);

    % --- SRP scores on theta-grid ---
    score = srp_scores_theta_v3(theta_grid, gcc_curves, lag_vec, dx_list, c, w_geom, coh_avg_list, srp_q);

    % --- Get Top-L separated candidates ---
    cand_thetas = pick_topL_theta_v3(theta_grid, score, srpTopL, srpMinSepDeg);
    if isempty(cand_thetas)
        [~, imax] = max(score);
        doa = theta_grid(imax);
        return;
    end

    bestCost  = inf;
    bestTheta = cand_thetas(1);

    for ci = 1:numel(cand_thetas)
        theta0 = cand_thetas(ci);

        theta_hat = theta0;
        u_hat = cosd(theta_hat);
        u_meas_last = [];
        w_meas_last = [];

        for it = 1:guided_iters
            u0 = cosd(theta_hat);

            % Adaptive tau window: broaden around broadside (u~0), tighten near end-fire (|u|~1)
            tau_win_ratio_eff = tau_win_ratio * (1 + tauWinGain * (1 - abs(u0)));

            [u_meas, w_meas] = guided_pick_and_weight( ...
                gcc_curves, lag_vec, dx_list, c, u0, ...
                w_geom, coh_avg_list, tau_win_ratio_eff, psr_delta, psr_guard_ratio);

            if isempty(u_meas)
                break;
            end

            % MAD outlier rejection before fusion (prevents single bad pair dominating)
            if numel(u_meas) >= 3
                u_med = median(u_meas);
                r = u_meas - u_med;
                madv = median(abs(r)) + 1e-12;

                k_mad = madBaseK - madTighten*abs(u0);   % slightly stricter near end-fire
                k_mad = max(2.0, min(3.5, k_mad));

                keep = abs(r) <= k_mad * madv;
                if sum(keep) >= madMinRemain && sum(keep) < numel(u_meas)
                    u_meas = u_meas(keep);
                    w_meas = w_meas(keep);
                end
            end

            u_hat = irls_scalar_huber(u_meas, w_meas, huber_k_u, irls_iter);
            theta_hat = acosd(max(min(u_hat,1),-1));

            u_meas_last = u_meas;
            w_meas_last = w_meas;
        end

        if isempty(u_meas_last)
            % fallback: if guided failed, use SRP init and give it a high cost
            theta_hat = theta0;
            u_hat = cosd(theta_hat);
            cost = inf;
        else
            cost = huber_cost_scalar_v3(u_meas_last, w_meas_last, u_hat, huber_k_u);
        end

        if cost < bestCost
            bestCost  = cost;
            bestTheta = theta_hat;
        end
    end

    doa = bestTheta;
end

function score = srp_scores_theta_v3(theta_grid, gcc_curves, lag_vec, dx_list, c, w_geom, coh_avg, q)
% Compute SRP-like score on theta-grid using GCC curve value at predicted tau
    P = numel(dx_list);
    score = zeros(numel(theta_grid),1);
    epsv = 1e-12;

    gcc_abs = abs(gcc_curves);

    for i = 1:numel(theta_grid)
        u = cosd(theta_grid(i));
        s = 0;
        for p = 1:P
            dx = dx_list(p);
            if abs(dx) < 1e-12, continue; end
            tau = dx*u/c;
            g = interp1(lag_vec, gcc_abs(:,p), tau, 'linear', 0);
            wp = w_geom(p) * (coh_avg(p) + epsv);
            s = s + wp * (g^q);
        end
        score(i) = s;
    end
end

function cand = pick_topL_theta_v3(theta_grid, score, L, minSepDeg)
% Pick top-L peaks with non-maximum suppression in theta domain
    cand = [];
    if isempty(theta_grid), return; end

    [~, ord] = sort(score, 'descend');
    ord = ord(:).';

    for k = 1:numel(ord)
        th = theta_grid(ord(k));
        if isempty(cand)
            cand = th;
        else
            if all(abs(cand - th) >= minSepDeg)
                cand(end+1,1) = th; %#ok<AGROW>
            end
        end
        if numel(cand) >= L
            break;
        end
    end

    % ensure column
    cand = cand(:);
end

function J = huber_cost_scalar_v3(us, ws, u, k)
% Weighted Huber cost for scalar fusion
    r = u - us;
    ar = abs(r);
    idx = ar <= k;
    J = sum(ws(idx) .* (0.5 * r(idx).^2)) + sum(ws(~idx) .* (k*(ar(~idx) - 0.5*k)));
end

function [gcc_curves, lag_vec, dx_list, w_geom, coh_avg_list] = ...
    build_gcc_curves_v3(Yuse, rmic, c, fs, win_size, f_vect, start_f, end_f, ...
                        os, beta, freq_alpha, coh_alpha, geom_eta)

% Frame-weighted cross-spectrum estimation + GCC
% Weight frames by energy and frame-level MSC (across frequency)

    [nfrms, M, ~] = size(Yuse);
    pairs = nchoosek(1:M,2);
    P = size(pairs,1);

    dx_list = zeros(P,1);
    d_list  = zeros(P,1);
    for p = 1:P
        m1 = pairs(p,1); m2 = pairs(p,2);
        dx_list(p) = rmic(m1,1) - rmic(m2,1);
        d_list(p)  = norm(rmic(m1,:) - rmic(m2,:));
    end

    d_norm = d_list / max(d_list);
    w_geom = (d_norm .^ geom_eta);     % <-- key change: soften long baseline dominance

    N  = win_size;
    N2 = N * os;
    lag_vec = (-N2/2 : N2/2-1).' / (fs*os);
    gcc_curves = zeros(numel(lag_vec), P);
    coh_avg_list = zeros(P,1);

    epsv = 1e-12;
    fmax = f_vect(end_f);

    % ---------------- Frame weights ----------------
    ref = 1;
    E = squeeze(sum(abs(Yuse(:,ref,start_f:end_f)).^2, 3));     % [nfrms,1]
    Em = median(E) + epsv;
    En = min(E/Em, 4);                                         % cap
    En = En .^ 0.7;                                            % energy exponent

    % Frame MSC: average across pairs (frequency-consistency in that frame)
    msc = zeros(nfrms,1);
    for t = 1:nfrms
        s = 0;
        for pp = 1:P
            m1 = pairs(pp,1); m2 = pairs(pp,2);
            X1 = squeeze(Yuse(t,m1,start_f:end_f)).';
            X2 = squeeze(Yuse(t,m2,start_f:end_f)).';
            P12 = sum(X1 .* conj(X2));
            P11 = sum(abs(X1).^2);
            P22 = sum(abs(X2).^2);
            s = s + (abs(P12)^2 / (P11*P22 + epsv));
        end
        msc(t) = s / P;
    end
    msc = min(max(msc, 0), 1);
    Wt  = (msc .^ 2.0) .* En;                                  % msc exponent is important
    Wt  = Wt / (sum(Wt) + epsv);

    % ---------------- Per-pair GCC curve ----------------
    for p = 1:P
        m1 = pairs(p,1); m2 = pairs(p,2);

        P12_pos = zeros(N/2+1, 1);
        coh_sum = 0; coh_cnt = 0;

        for k = start_f:end_f
            f = f_vect(k);
            wf = (f/fmax)^freq_alpha;

            X1 = Yuse(:,m1,k);
            X2 = Yuse(:,m2,k);

            % weighted cross/auto spectra across frames
            P12 = sum(Wt .* (X1 .* conj(X2)));
            P11 = sum(Wt .* (abs(X1).^2));
            P22 = sum(Wt .* (abs(X2).^2));

            coh = abs(P12)^2 / (P11*P22 + epsv);
            coh = min(max(coh,0),1);

            P12_phat = P12 / ((abs(P12)+epsv)^beta);
            P12_pos(k) = wf * (coh^coh_alpha) * P12_phat;

            coh_sum = coh_sum + coh;
            coh_cnt = coh_cnt + 1;
        end

        coh_avg_list(p) = coh_sum / max(coh_cnt,1);

        P12_full = [P12_pos; conj(P12_pos(end-1:-1:2))];

        Xs  = fftshift(P12_full);
        pad = N2 - N;
        Xs2 = [zeros(floor(pad/2),1); Xs; zeros(ceil(pad/2),1)];
        X2  = ifftshift(Xs2);

        r = real(ifft(X2));
        r = fftshift(r);

        % normalize per-pair to prevent scale differences
        r = r / (max(abs(r)) + epsv);
        gcc_curves(:,p) = r;
    end
end

% ======================================================================

function theta0 = srp_search_theta_v3(theta_grid, gcc_curves, lag_vec, dx_list, c, w_geom, coh_avg, q, clip_factor)
% Robust SRP accumulation: winsorize pair contributions to prevent domination
    P = numel(dx_list);
    score = zeros(numel(theta_grid),1);
    epsv = 1e-12;

    for i = 1:numel(theta_grid)
        u  = cosd(theta_grid(i));

        cp = zeros(P,1);
        for p = 1:P
            dx = dx_list(p);
            if abs(dx) < 1e-12, continue; end
            tau = dx*u/c;

            g = interp1(lag_vec, gcc_curves(:,p), tau, 'linear', 0);
            g = abs(g);

            wp = w_geom(p) * (coh_avg(p) + epsv);
            cp(p) = wp * (g^q);
        end

        medc = median(cp(cp>0));
        if isempty(medc) || medc<=0, medc = epsv; end
        cp = min(cp, clip_factor*medc);

        score(i) = sum(cp);
    end

    [~, imax] = max(score);
    theta0 = theta_grid(imax);

    % parabolic refine
    if imax>1 && imax<numel(theta_grid)
        y1 = score(imax-1); y2 = score(imax); y3 = score(imax+1);
        denom = (y1 - 2*y2 + y3);
        if abs(denom) > 1e-12
            d = 0.5*(y1 - y3)/denom;
            step = min(theta_grid(imax)-theta_grid(imax-1), theta_grid(imax+1)-theta_grid(imax));
            theta0 = max(min(theta0 + d*step, 180), 0);
        end
    end
end

% ======================================================================

function [u_meas, w_meas] = guided_pick_and_weight_v3( ...
    gcc_curves, lag_vec, dx_list, c, u0, w_geom, coh_avg, ...
    tau_win_ratio, tau_sigma_ratio, psr_delta, psr_guard_ratio, ...
    endfire_damp, endfire_pow, K_local)

% Pick peak by composite score:
% score = |r(tau)| * PSR^delta * exp(- (tau-tau_pred)^2 / (2*sigma^2))

    P = numel(dx_list);
    epsv = 1e-12;
    dt = lag_vec(2) - lag_vec(1);

    dx_abs_max = max(abs(dx_list)) + epsv;

    u_meas = [];
    w_meas = [];

    for p = 1:P
        dx = dx_list(p);
        if abs(dx) < 1e-12, continue; end

        tau_pred = dx*u0/c;
        tau_max  = abs(dx)/c;

        tau_win = max(2*dt, tau_win_ratio*tau_max);

        % restrict to feasible range
        lo = max(-tau_max, tau_pred - tau_win);
        hi = min( tau_max, tau_pred + tau_win);

        idx = find(lag_vec >= lo & lag_vec <= hi);
        if numel(idx) < 7, continue; end

        r = gcc_curves(:,p);
        a = abs(r(idx));

        % find local maxima inside idx (no toolbox)
        cand_local = find( a(2:end-1) >= a(1:end-2) & a(2:end-1) > a(3:end) ) + 1;
        if isempty(cand_local)
            [~, im] = max(a);
            cand_local = im;
        end

        % keep top-K_local by amplitude
        [~, ord] = sort(a(cand_local), 'descend');
        cand_local = cand_local(ord);
        cand_local = cand_local(1:min(K_local, numel(cand_local)));

        bestScore = -inf;
        bestTau   = lag_vec(idx(cand_local(1)));
        bestPSR   = 1;

        sigma = max(2*dt, tau_sigma_ratio*tau_win);

        for ii = 1:numel(cand_local)
            k0 = idx(cand_local(ii));
            tau_hat = lag_vec(k0);

            % parabolic refine on abs
            if k0>1 && k0<numel(lag_vec)
                y1 = abs(r(k0-1)); y2 = abs(r(k0)); y3 = abs(r(k0+1));
                denom = (y1 - 2*y2 + y3);
                if abs(denom) > 1e-12
                    d = 0.5*(y1 - y3)/denom;
                    tau_hat = tau_hat + d*dt;
                end
            end

            % PSR computed within feasible region, excluding guard around peak
            guard = psr_guard_ratio*tau_max;
            feas  = (lag_vec >= -tau_max) & (lag_vec <= tau_max);
            mask  = feas & ((lag_vec < (tau_hat-guard)) | (lag_vec > (tau_hat+guard)));
            side  = median(abs(r(mask)));
            peakv = abs(interp1(lag_vec, abs(r), tau_hat, 'linear', abs(r(k0))));
            psr   = peakv / (side + epsv);
            psr   = min(max(psr,1),100);

            clos  = exp(-0.5*((tau_hat - tau_pred)/sigma)^2);
            sc    = peakv * (psr^psr_delta) * clos;

            if sc > bestScore
                bestScore = sc;
                bestTau   = tau_hat;
                bestPSR   = psr;
            end
        end

        u_p = (c * bestTau) / dx;
        u_p = max(min(u_p,1),-1);

        % end-fire adaptive damping for long baselines
        dnorm = abs(dx) / dx_abs_max; % 0~1
        w_end = 1 / (1 + endfire_damp * (abs(u0)^endfire_pow) * (dnorm^2));

        w_p = (w_geom(p) * (coh_avg(p) + epsv) * w_end) * (bestPSR^psr_delta);

        u_meas(end+1,1) = u_p; %#ok<AGROW>
        w_meas(end+1,1) = w_p; %#ok<AGROW>
    end
end

function c = speed_of_sound(T_celsius, RH_percent, P_kPa)
% Speed of sound in moist air using Cramer (1993) approximation.
% T_celsius: temperature (C), RH_percent: relative humidity (%), P_kPa: pressure (kPa)
    T = T_celsius + 273.15;
    h = max(min(RH_percent, 100), 0) / 100; % clamp to [0,1]
    P = P_kPa * 10; % kPa -> mbar (hPa)

    % Saturation vapor pressure (mbar) using Buck (1981)
    e_s = 6.1121 * exp((18.678 - T_celsius/234.5) * (T_celsius/(257.14 + T_celsius)));
    e = h * e_s; % partial pressure of water vapor (mbar)

    x_w = e / P; % molar fraction of water vapor
    c = 331.45 * sqrt(1 + (T_celsius / 273.15)) * (1 + 0.51 * x_w);
end