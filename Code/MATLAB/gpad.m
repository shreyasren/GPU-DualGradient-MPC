clear all
close all

n = 3; % this is the number of battery cells
p = 4; % this is the number of steps in prediction horizon
numSamples = 1000;

% Initialize the state-of-charge of all cells
if n == 10
    x0 = [-0.1 0.45 -0.09 0.05 0 -0.05 0.3 0.2 0.25 -0.45]';
elseif n == 5
    x0 = [-0.1 0.05 0 -0.05 0.1]';
else
    x0 = (rand(1, n) - 0.5)';
end 

% Cell capacities in Ah
CellCapacities = 0.027*4.1*ones(1, n);
x0_t = x0;
x = zeros(n, numSamples);
x_exp = zeros(n, numSamples);
u_exp = zeros(n, numSamples);
u = zeros(n, numSamples);
timeQPSolve = zeros(numSamples, 1);

% Box constraints:
% xmin and xmax constrain the SOC to be bewteen -0.5 and 0.5
% umin and umax constrain the balancing currents to be 
% between -0.3 and 0.3 Amps. 
xmin(1:n*p, 1) = -0.5;
xmax(1:n*p, 1) = 0.5;
umin(1:n*p, 1) = -0.3;
umax(1:n*p, 1) = 0.3;
A = eye(n);
B = zeros(n);
M_ab = zeros(n*p, n*p);
M_ak = zeros(n*p, n);
qx_weight = 100; %10
qu_weight = 1; %1
Qx = qx_weight*eye(n);
Qu = qu_weight*eye(n);
Mx = qx_weight*eye(n*p);
Mu = qu_weight*eye(n*p);
K = zeros(p, n*p);

% populate B and M_ak
for i = 1:n
    B(i, i) = -1/(3600*CellCapacities(i));
end
for i = 1:p
    M_ak((i-1)*n+1:(i-1)*n+n, 1:n) = A^i;
end

% populate M_ab
for i = 1:p
    for j = 1:p
        if (j > i)
            M_ab((i-1)*n+1:(i-1)*n+n, (j-1)*n+1:(j-1)*n+n) = 0;
        else
            M_ab((i-1)*n+1:(i-1)*n+n, (j-1)*n+1:(j-1)*n+n) = (A^(i-j))*B;
        end
    end
end

for i = 1:p
    for j = 1:n*p
        if floor((j-1)/n) + 1 == i
            K(i, j) = 1;
        else
            K(i, j) = 0;
        end
    end
end

% Hessian and linear matrices as they appear in OCP objective function
H = M_ab'*Mx*M_ab + Mu;
F = M_ak'*Mx*M_ab;

for i = 1:numSamples
    x(:,i) = x0_t;
    f = x0_t'*F;

    % Equality and inequality matrices, extracted from constraints
    A_i = [M_ab; -1*M_ab; eye(n*p); -1*eye(n*p); K; -1*K];
    b_i = [xmax-M_ak*x0_t; -xmin+M_ak*x0_t; umax; -umin; zeros(p, 1); zeros(p, 1)];
    
    % GPAD algorithm -- we can enable the built-in quadprog as a benchmark
    tic;
    % u_mpc_traj = quadprog(H, f, A_i, b_i, [], []);
    [u_mpc_traj, avg_alg_times] = acceldualgrad(H, f, A_i, b_i, Qx, Qu, n);
    u_mpc = u_mpc_traj(1:n, 1);
    timeQPSolve(i) = toc;
    x0_t = A*x0_t + B*u_mpc;
    u(:,i) = u_mpc;
end

% Plot simulation results
subplot(1,2,1)
plot(1:1:numSamples, x, 'LineWidth', 2)
title('Cell Balancing with MPC', 'FontSize', 18)
xlabel('Time [min]', 'FontSize', 16)
ylabel('State of Charge', 'FontSize', 16)
legend('Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', 'Cell 5', 'Cell 6', 'Cell 7', 'Cell 8', 'Cell 9', 'Cell 10', 'FontSize', 12)
grid on
grid minor

subplot(1,2,2)
plot(1:1:numSamples, u, 'LineWidth', 2)
title('Balancing Currents', 'FontSize', 18)
xlabel('Time [min]', 'FontSize', 16)
ylabel('Current (A)', 'FontSize', 16)
legend('Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', 'Cell 5', 'Cell 6', 'Cell 7', 'Cell 8', 'Cell 9', 'Cell 10', 'FontSize', 12)
grid on
grid minor