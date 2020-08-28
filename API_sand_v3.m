%%

function [y p, k_initial1] = API_sand_v3(water, D, gamma, phi_deg, Z, cyclic, xmin, xmax)
% API sand pyĿ�� ����

% API_sand_v3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% v3 (2020-05-20)
% phi = 25���� ���� k_initial �� �߰���. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Water: 0�̸� water table ��, 1�̸� water table �Ʒ�
% D: ���� ���� (m)
% gamma: �����߷� (kN/m^3)
% phi_deg: ���θ����� (degree) 
% Z: py ������ ���� (m)
% cycllic: 0�̸� static loading, 1�̸� cyclic loading
% x��(p-y��� y��) ��������

% �Է� ����:
% D = 0.7; % m
% gamma = 18; % kN/m^3
% phi_deg = 30; % degree
% Z = 2; % m
% cyclic = 0; % cyclic �̸� 1

phi = phi_deg*(pi/180); % rad

% A�� ���ϱ� (A: factor to account for cyclic or static loading condition)
if cyclic == 0
    A(1) = (3.0 - 0.8*Z/D);
    A(2) = 0.9;
    A = max(A);
elseif cyclic  == 1
    A = 0.9;
end

% k(initial, kN/m3) ���ϱ�

% ���� ��
phi_1 = [25 30 35 40]; % degree
k_1 = [5400 11000 22000 45000]; % kN/m^3
phi_11 = [25:0.01:40];
k_22 = interp1(phi_1, k_1, phi_11, 'spline');
kp1_lit = interp1(phi_1, k_1, phi_deg, 'spline');

if water == 0
    % ���ϼ��� �� (API) (API �׷����κ��� ���� ��. phi = 29�� ���ϴ� ����ġ)
    phi_aw = [25 29 30 36 40];
    k_aw = [3000 4035 11567 43040 73975]; 
    phi_22 = [29:0.01:40];
    k_aww = interp1(phi_aw, k_aw, phi_22);
    kp2_aw = interp1(phi_aw, k_aw, phi_deg);
    k_initial1 = kp2_aw
    
    
elseif water == 1
    % ���ϼ��� �Ʒ� (API) (API �׷����κ��� ���� ��. phi = 29�� ���ϴ� ����ġ)
    phi_uw = [25 29 30 36 40];
    k_uw = [3000 4035 8877 24210 41695];
    phi_33 = [29:0.01:40];
    k_uww = interp1(phi_uw, k_uw, phi_33);
    kp3_uw = interp1(phi_uw, k_uw, phi_deg);
    k_initial1 = kp3_uw
    % k_initial1 = 1500
    
    
end

% % ���� �� �Ǿ����� Ȯ�� �׷��� plot
% figure(101)
% plot(phi_1, k_1, 'o', phi_11, k_22, 'DisplayName','lit');
% hold on
% if water == 0
%     plot(phi_aw, k_aw, 'x', phi_22, k_aww, 'DisplayName','API_{aw}');
% else
%     plot(phi_uw, k_uw, 'x', phi_33, k_uww, 'DisplayName','API_{uw}');
% end
% hold off
% xlabel('\phi (degree)');
% ylabel('kN/m^3');
% legend

% C1, C2, C3 �� ���ϱ�. �Ʒ� �� 1�� 2�� �� �����ؼ� �ؾ���. 2��° ���� �´µ�?

% ��1.(������: Rocscience 2018)
% alpha = phi/2;
% beta = pi/4 + phi/2;
% K0 = 0.4;
% Ka = (1-sin(phi))/(1+sin(phi));
% C1 = (tan(beta)^2*tan(alpha)/tan(beta-phi))+K0*((tan(phi)*tan(beta))/(cos(alpha)*tan(beta-phi))+tan(beta)*(tan(phi)*sin(beta)-tan(alpha)));
% C2 = tan(beta)/tan(beta-phi)+Ka;
% C3 = Ka*(tan(beta)^8-1)+K0*tan(phi)*tan(beta)^4;

% ��2. (������: Amarbouzid 2018)
C1 = 0.115*10^(0.0405*phi_deg);
C2 = 0.571*10^(0.022*phi_deg);
C3 = 0.646*10^(0.0555*phi_deg);


% pu(kN/m) ���ϱ�
pu(1) = (C1*Z+C2*D)*gamma*Z;
pu(2) = C3*D*gamma*Z;
pu = min(pu);

% py curve
y = xmin:0.001:xmax; % y plot range
p = A*pu*tanh((k_initial1*Z*y/(A*pu)));
figure(102)
plot(y,p)
hold on
xlabel('Pile deflection, m');
ylabel('Soil reaction, kN/m');
title('API py-curve for sand');
xlim([xmin xmax]);

end
% % Plaxis py curve
% y_plaxis = [0 0.003478885	0.008536755	0.025524132	0.07225538	0.153799406];
% p_plaxis = [0 28.685	59.5461	121.0804	204.2874	283.9796];
% plot(y_plaxis, p_plaxis);
% hold off
