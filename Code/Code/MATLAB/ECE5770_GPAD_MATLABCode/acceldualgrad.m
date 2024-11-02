function [u, avg_alg_times] = acceldualgrad(H, f, A_i, b_i, Qx, Qu, n_u)
%ACCELDUALGRAD Implements the accelerated dual gradient-projection algorithm
% -- per Patrinos' and Bemporad's paper -- to solve the QP problem
% produced by the MPC formulation. 

num_iterations = 100; 

% Sizes and constants
m = size(A_i, 1); % number of constraints
n = size(H, 2); % number of controls over prediction horizon
L = norm(H, 'fro')^2; % Lipschitz constant
e_g = 1e-6;
e_V = 1e-6;

% Initial values
y_m1 = zeros(m, 1); y_0 = zeros(m, 1); 
z_m1 = zeros(n, 1);
th_m1 = 1; th_0 = 1; 
% Pre-computed matrices (can probably move this outside of fcn)
M_G = inv(H)*(A_i'); 
g_P = inv(H)*(f'); 
G_L = (1/L)*A_i; 
p_D = (-1/L)*b_i; 
% Registered variables
y_v = y_0; y_vm1 = y_m1; 
z_vm1 = z_m1; 
beta_v = 0;
th_v = th_0; th_vm1 = th_m1;
% Anonymous helper functions for Lagrangian, primal, and dual functions
valuefcn = @(z) (0.5*z'*H + f)*z;
lagrangian = @(z, y) (0.5*z'*H + f + y'*A_i)*z - y'*b_i; 
dualtoprimal = @(y) -1*inv(H)*(f' + A_i'*y);
dualfcn = @(y) lagrangian(dualtoprimal(y), y); 
g = @(z) A_i*z - b_i;

% Algorithm 1: Accelerated Dual Projection Algorithm
alg_times = []; 
% while (1)
for i=1:100

    alg_time = zeros(1, 5);
    tic
    w_v = y_v + beta_v*(y_v - y_vm1); % (8a)
    alg_time(1) = toc; 
    tic
    zhat_v = -1*M_G*w_v - g_P; % (8b)
    alg_time(2) = toc; 
    tic
    z_v = (1 - th_v)*z_vm1 + th_v*zhat_v; % (8c)
    alg_time(3) = toc; 
    tic
    y_vp1 = max(w_v + G_L*zhat_v + p_D, 0); % (8d)
    alg_time(4) = toc; 
    tic
    th_vp1 = (sqrt(th_v^4 + 4*th_v^2) - th_v^2)/2; % (8e)
    beta_v = th_v*((1/th_vm1) - 1); 
    alg_time(5) = toc; 
    alg_times = [alg_times; alg_time];
    % Shifting registered variables
    y_vm1 = y_v; 
    y_v = y_vp1; 
    z_vm1 = z_v; 
    th_vm1 = th_v; 
    th_v = th_vp1; 

    % Algorithm 2: Terminating criterion
    % if max(g(z_v), 0) <= e_g
    %     break
    % elseif max(g(zhat_v), 0) <= e_g
    %     if w_v >= 0
    %         if -1*w_v'*g(zhat_v) <= e_V
    %             break
    %         elseif -1*w_v'*g(zhat_v) <= valuefcn(zhat_v)*e_V/(1+e_V)
    %             break
    %         end
    %     elseif valuefcn(zhat_v) - dualfcn(y_vp1) <= e_V*max(dualfcn(y_vp1), 1)
    %         break
    %     end  
    % end
end

% z_opt = -1*inv(H)*(f' + A_i'*y_v);
u = z_v(1:n_u);
avg_alg_times = mean([alg_times; alg_times]); 
end

