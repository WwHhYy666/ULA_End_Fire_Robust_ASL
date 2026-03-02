clear; clc;

%% ===================== Config =====================
mat_file = "D:\acoustic\asl\EXPERIMENT\End-fire\DOA_results_1m.mat"; % change to 2m if needed
th_deg = 6;
endfire_angles = [20, 30, 150, 160];

%% ===================== Load =====================
if ~exist(mat_file, "file")
    error("MAT file not found: %s", mat_file);
end

data = load(mat_file);

% Accept either top-level fields or a single struct/table variable.
if isfield(data, "TrueAngle")
    S = data;
else
    fns = fieldnames(data);
    if numel(fns) == 1
        S = data.(fns{1});
    else
        error("Unsupported MAT structure. Expected TrueAngle fields or a single struct variable.");
    end
end

true_angle = get_field(S, "TrueAngle");

alg_names = ["MVDR", "SRP-PHAT", "W-SRP-PHAT", "GCC-WLS"];
field_names = ["Est_MVDR", "Est_SRP_PHAT", "Est_W_SRP_PHAT", "Est_GCCWLS"];

nA = numel(alg_names);

rmse_all = nan(nA, 1);
acc_all = nan(nA, 1);
rmse_end = nan(nA, 1);

for k = 1:nA
    est = get_field(S, field_names(k));
    err = est - true_angle;

    rmse_all(k) = sqrt(mean(err.^2, "omitnan"));
    acc_all(k) = mean(abs(err) <= th_deg, "omitnan");

    mask_end = ismember(true_angle, endfire_angles);
    err_end = err(mask_end);
    rmse_end(k) = sqrt(mean(err_end.^2, "omitnan"));
end

%% ===================== Report =====================
fprintf("==== Global RMSE and ACC@%d deg ====%s", th_deg, newline);
for k = 1:nA
    fprintf("%-12s | RMSE: %.4f | ACC: %.4f%s", alg_names(k), rmse_all(k), acc_all(k), newline);
end

fprintf("%s==== End-fire RMSE (20/30/150/160) ====%s", newline, newline);
for k = 1:nA
    fprintf("%-12s | RMSE: %.4f%s", alg_names(k), rmse_end(k), newline);
end

%% ===================== Helper =====================
function v = get_field(S, name)
    if isstruct(S)
        if ~isfield(S, name)
            error("Missing field: %s", name);
        end
        v = S.(name);
    elseif istable(S)
        if ~ismember(name, S.Properties.VariableNames)
            error("Missing table column: %s", name);
        end
        v = S.(name);
    else
        error("Unsupported data type for metrics.");
    end

    v = v(:);
end
