%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2020-05-27
% q_from_M_v7_VolumePile
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

% Soil parameter (py ���)
Water =1;
gamma = 18-10; % ��ȿ �����߷�
phi = 29;
pydepth = -1; % py curve ���� ����

% Pile parameter 
PileLength = 10; % �Ʒ� for�� �ȿ��� Pile_L�̶�� ������ ����.
D = 0.5; % Pile diameter
Ansys = 10; % �ؼ� ����Ʈ ����
PileNum = 9; % �������� ����

tol1 = 0.00001; % py curve ���� ������ z index ã���� tolerance
RigidityFactor = 10^6; % Volume Pile �߰��� ���Ե� Beam����� ���ҵ� ������ ���

% Plaxis data �ҷ�����
FileName_force = ...
    ["3.1.2.1.1 Calculation results_ Beam_ Phase_1 [Phase_1] (1_34)_ Table of forces"...
    ,"3.1.2.1.2 Calculation results_ Beam_ Phase_2 [Phase_2] (2_58)_ Table of forces"...
    ,"3.1.2.1.3 Calculation results_ Beam_ Phase_3 [Phase_3] (3_91)_ Table of forces"...
    ,"3.1.2.1.4 Calculation results_ Beam_ Phase_4 [Phase_4] (4_119)_ Table of forces"...
    ,"3.1.2.1.5 Calculation results_ Beam_ Phase_5 [Phase_5] (5_146)_ Table of forces"...
    ,"3.1.2.1.6 Calculation results_ Beam_ Phase_6 [Phase_6] (6_171)_ Table of forces"...
    ,"3.1.2.1.7 Calculation results_ Beam_ Phase_7 [Phase_7] (7_195)_ Table of forces"...
    ,"3.1.2.1.8 Calculation results_ Beam_ Phase_8 [Phase_8] (8_223)_ Table of forces"...
    ,"3.1.2.1.9 Calculation results_ Beam_ Phase_9 [Phase_9] (9_250)_ Table of forces"...
    ,"3.1.2.1.10 Calculation results_ Beam_ Phase_10 [Phase_10] (10_288)_ Table of forces"];
FileName_disp = ...
    ["3.1.1.1.1.1 Calculation results_ Beam_ Phase_1 [Phase_1] (1_34)_ Table of total displacements"...
    ,"3.1.1.1.1.2 Calculation results_ Beam_ Phase_2 [Phase_2] (2_58)_ Table of total displacements"...
    ,"3.1.1.1.1.3 Calculation results_ Beam_ Phase_3 [Phase_3] (3_91)_ Table of total displacements"...
    ,"3.1.1.1.1.4 Calculation results_ Beam_ Phase_4 [Phase_4] (4_119)_ Table of total displacements"...
    ,"3.1.1.1.1.5 Calculation results_ Beam_ Phase_5 [Phase_5] (5_146)_ Table of total displacements"...
    ,"3.1.1.1.1.6 Calculation results_ Beam_ Phase_6 [Phase_6] (6_171)_ Table of total displacements"...
    ,"3.1.1.1.1.7 Calculation results_ Beam_ Phase_7 [Phase_7] (7_195)_ Table of total displacements"...
    ,"3.1.1.1.1.8 Calculation results_ Beam_ Phase_8 [Phase_8] (8_223)_ Table of total displacements"...
    ,"3.1.1.1.1.9 Calculation results_ Beam_ Phase_9 [Phase_9] (9_250)_ Table of total displacements"...
    ,"3.1.1.1.1.10 Calculation results_ Beam_ Phase_10 [Phase_10] (10_288)_ Table of total displacements"];

% zcol = [2, 169]; % �������� �� �ش� ������ column ��ǥ (�Ŵ��� ����)
% ���� ���� ����(�ӵ� ������ �ϱ� ����)
F_input_diff = zeros(Ansys, PileNum);
F_input_fit = zeros(Ansys, PileNum);
F_input_intp = zeros(Ansys, PileNum);
qf_out = zeros(Ansys+1, PileNum);
zf_out = zeros(Ansys+1, PileNum);
qint_out = zeros(Ansys+1, PileNum);
zint_out = zeros(Ansys+1, PileNum);

SampleData = importfile_force(FileName_force(1)); % �ش� ������ column��ǥ ã������ �ؼ������� �� �ϳ��� ���÷� �ҷ���
zz = SampleData(2:end,6); % z ��ǥ�� �ҷ���
zz = table2array(zz); % �ҷ��� ���� double ������ �ٲ�
[zzsize_m, zzsize_n] = size(zz); % zz�� ������ ��ȯ
DataSize = zzsize_m/PileNum; % �ϳ��� pile �����Ϳ� ���� data column ������ ��ȯ

% [embedded beam ����] ---------------------------------------------------
%  �������� �� �ش� ������ column ��ǥ ã�� (���߿� �̰� Ȱ���ؼ� �߰� �ڵ� �ʿ�)
% SampleData = importfile_force(FileName_force(1)); % �ش� ������ column��ǥ ã������ �ؼ������� �� �ϳ��� ���÷� �ҷ���
% zz = SampleData(1:end,6); % z ��ǥ�� �ҷ���
% zz = table2array(zz); % �ҷ��� ���� double ������ �ٲ�
% zcol(:,1)=find(zz<=0.00000001 & zz>-0.00000001, PileNum); % z ��ǥ�� 0�� ���� column index�� �ҷ���
% zcol(:,2)=find(zz==-PileLength, PileNum); % z ��ǥ�� 10�� ���� column index�� �ҷ���
% [zcol_size_m, zcol_size_n]=size(zcol);
% DataColSize = zcol(:,2) - zcol(:,1) + 1;
% zarray = zeros(max(DataColSize), zcol_size_m);
% Qarray = zeros(max(DataColSize), zcol_size_m);
% Marray = zeros(max(DataColSize), zcol_size_m);
% Dispxarray = zeros(max(DataColSize), zcol_size_m);
% -----------------------------------------------------------------------

zcol = zeros(PileNum, 2);
for LL = 1: PileNum
    zcol(LL,1) = 2+(LL-1)*DataSize;
    zcol(LL,2) = DataSize+1+(LL-1)*DataSize;
end

% ���� ���� ����(�ӵ� ������ �ϱ� ����)
zarray = zeros(DataSize, PileNum);
Qarray = zeros(DataSize, PileNum);
Marray = zeros(DataSize, PileNum);
Dispxarray = zeros(DataSize, PileNum);

for i = 1:Ansys
    
    % ������ �ҷ�����
    Data_Force = importfile_force(FileName_force(i));
    Data_Disp = importfile_disp(FileName_disp(i));
        
    for j = 1:PileNum
        
        Data_Force_PileNum = Data_Force(zcol(j,1):zcol(j,2),1:end); % �������� �� j��° ���ҿ� ���� Force ������
        Data_Force_PileNum = sortrows(Data_Force_PileNum, 6, 'descend'); % �����͸� ���̼����� ����
        z = Data_Force_PileNum(:,6); % Data_Force_PileNum���� z���� �ҷ���
        zarray(1:size(table2array(z)),j) = table2array(z); % �������� j�� ���� j���� z�� ����
        Pile_L(:,j) = min(zarray(:,j)); % Pile�� ���� ��ȯ (�������� j���� ���� ���� z��)
        Q = Data_Force_PileNum(:,10); % Data_Force_PileNum���� Q���� �ҷ���
        Qarray(1:size(table2array(Q)),j) = table2array(Q)*RigidityFactor; % �������� j�� ���� j���� Q�� ����
        M = Data_Force_PileNum(:,19); % Data_Force_PileNum���� M���� �ҷ���
        Marray(1:size(table2array(M)),j) = table2array(M)*RigidityFactor; % �������� j�� ���� j���� M�� ����
        
        Data_Disp_PileNum = Data_Disp(zcol(j,1):zcol(j,2),1:end); % �������� �� j��° ���ҿ� ���� Disp ������
        Data_Disp_PileNum = sortrows(Data_Disp_PileNum, 6, 'descend'); % �����͸� ���̼����� ����
        Dispx =  Data_Disp_PileNum(:, 7); % Data_Disp_PileNum���� Dispx���� �ҷ���
        Dispxarray(1:size(table2array(Dispx)),j) = table2array(Dispx); % �������� j�� ���� j���� dispx�� ����
        
        
        %%% ���1. ���׽����� M�� �������� �ʰ� ��ġ�̺�(����)�Ͽ� Q�� q����
        dz = diff(zarray(:,j)); % m
        dM = diff(Marray(:,j)); % kN-m
        Q_diff = dM./dz; % kN
        dQ_diff = diff(Q_diff); % kN
        q_diff = dQ_diff./dz(2:end); % kn/m
        
        
        %%% ���2. ���׽����� M������ �����ϰ� �̸� �̺��Ͽ� Q�� q����
        polyfit_M = polyfitB(zarray(:,j),Marray(:,j), 6, 0); % z�� ���� M�� ���׽� ����
        % polyfitB�� Ư�� y������ ������ fitting�� ����� �Լ�
        % polyfitB(x, y, ����, y����)
        % https://kr.mathworks.com/matlabcentral/fileexchange/35401-polyfitzero ����
        % ������ ������ polyfit��
        % https://kr.mathworks.com/matlabcentral/fileexchange/54207-polyfix-x-y-n-xfix-yfix-xder-dydx
        
        % ���õ� Q ���׽� plot�� ���� ��������
        delta = 0.01;
        zf = 0:-delta:Pile_L;
        Mf = polyval(polyfit_M, zf);
        
        % Q ���׽��� �̺��Ͽ� Q (sheer force, kN), q (soil resistance, kN/m) ����
        polyfit_Q = polyder(polyfit_M);
        polyfit_q = polyder(polyfit_Q);
        Qf = polyval(polyfit_Q, zf);
        qf = polyval(polyfit_q, zf);
        
        
        %%% ���3. M data�� interplotation�ϰ�, �̸� ��ġ�̺��Ͽ� Q�� q����
        SampRate = 1; % PileLength������ index ã�� �� sampling rate (m)
        factor1 = 1 / SampRate; % PileLength���� sampling�� �����͸� �� ��(����)�� �����ϱ� ���� �۾�
        tol2 = 0.0001; % PileLength������ index�� ã�� �� ���� tolerance
        zarray_intp = zeros(PileLength * factor1 + 1, 1); % ù° z�� 0���� ��.
        Marray_intp = zeros(PileLength * factor1 + 1, 1); % ù° M�� 0���� ��.
        for nn = SampRate:SampRate:PileLength
            zcol2(nn*factor1,:)=find(zarray(:,j) >= -nn-tol2 & zarray(:,j) <= -nn+tol2, 1); % z = nn (m) PileLength������ index�� ã��
            zarray_intp(1+nn*factor1,:) = zarray(zcol2(nn*factor1),j); % 0 ���� z
            Marray_intp(1+nn*factor1,:) =Marray(zcol2(nn*factor1),j); % 0 ������ M
        end
        % �ռ� nn (m)���� sampling�� zarray_intp, Marray_intp�� zintp �������� ����
        zintp = [-PileLength:0.0001:0]; % interplotation�� z��ǥ
        Mintp = interp1(zarray_intp, Marray_intp, zintp, 'spline'); % zintp���� interplotation �ǽ�
        % M�� ��ġ�̺�
        dz_intp = diff(zintp); % m
        dM_intp = diff(Mintp); % kN-m
        Q_diff_intp = dM_intp./dz_intp; % kN
        dQ_diff_intp = diff(Q_diff_intp); % kN
        q_diff_intp = dQ_diff_intp./dz_intp(2:end); % kn/m
        
        
%         % plot
%         figure(j) % figure 1-j�� �� ���������� ��ġ
%         % M plot
%         subplot(3,Ansys,i);  % 1-3���� ���� Bending Moment, Shear forece, soil resistance; 1-i���� ���� �ܰ�
%         plot(zarray(:,j), Marray(:,j), 'o', zf, Mf, zintp, Mintp, 'k');
%         xlabel('z (m)')
%         ylabel('Moment (kN-m)')
%         legend('Plaxis', '���׽� fitting','Spline')
%         grid on
%         % Q plot
%         subplot(3,Ansys,i+Ansys)
%         plot(zarray(2:end,j), Q_diff, 'o', zf, Qf, zintp(2:end), Q_diff_intp,'k');
%         xlabel('z (m)')
%         ylabel('Shear force (kN)')
%         grid on
%         % q plot
%         subplot(3,Ansys,i+Ansys*2)
%         plot(zarray(3:end,j), q_diff, 'o', zf, qf, zintp(3:end), q_diff_intp,'k');  % ��ġ�̺а��� pit�Լ� �̺а�
%         xlabel('z (m)')
%         ylabel('Soil resistance (kN/m)')
%         ylim([-200*i 200*i])
%         grid on
%         f=figure(j);
%         f.Position = [30 80  1800 800];
        
        % % figure ����
        % pydepth_str = num2str(pydepth);
        % pydepth_str = strcat('_at_', pydepth_str, 'm');
        % PilePosition = ["Force_(D,D)","Force_(0,D)","Force_(-D,D)","Force_(D,0)","Force_(0,0)","Force_(-D,0)","Force_(D,-D)","Force_(0,-D)","Force_(-D,-D)"];
        % Figure_i_name = strcat(PilePosition, pydepth_str);
        % saveas(gcf, Figure_i_name(j), 'png')
                
        % ��ġ �����а��� �������� Ȯ��
        F_input_diff(i,j)=Qarray(1,j)-Qarray(end,j); % ���� head�� ������ ������ ������Ͽ� Ȯ�� (Plaxis ���)
        F_input_fit(i,j)=Qf(1)-Qf(end); % ���� head�� ������ ������ ������Ͽ� Ȯ�� (���׽� ���� ���)
        F_input_intp(i,j)=Q_diff_intp(1)-Q_diff_intp(end); % ���� head�� ������ ������ ������Ͽ� Ȯ�� (���׽� ���� ���)
        
        qf_out(1,j)=0; % (0,0)�� ���� �÷��ϱ� ���� ������ǥ ����
        zf_out(1,j)=0; % (0,0)�� ���� �÷��ϱ� ���� ������ǥ ����
        qf_out(i+1,j) = abs(polyval(polyfit_q,pydepth)); % pydepth �������� soil resistance�� ����
        zindex = find(zarray(:,j)>pydepth-tol1 & zarray(:,j)<pydepth+tol1, 1); % pydepth���� ���� ����� zindex�� ã��
        zf_out(i+1,j) = Dispxarray(min(zindex),j); % ã�� index�� �ش��ϴ� pile disp ���� ��ȯ
        
        qint_out(1,j)=0; % (0,0)�� ���� �÷��ϱ� ���� ������ǥ ����
        zint_out(1,j)=0; % (0,0)�� ���� �÷��ϱ� ���� ������ǥ ����
        z3 = zintp(3:end); % q_diff_intp �迭�� �����ϴ� z��
        pydepth_index1 = find(z3==pydepth,1); % z3 �� �� pydepth�� �ش��ϴ� index ����
        qint_out(i+1,j) = abs(q_diff_intp(pydepth_index1)); % pydepth �������� soil resistance�� ����
        pydepth_index2 = find(zarray(:,j)>pydepth-tol1 & zarray(:,j)<pydepth+tol1, 1); % pydepth���� ���� ����� zindex�� ã��
        zint_out(i+1,j) = Dispxarray(min(pydepth_index2),j);
    end
end

% % figure ����
% PilePosition = ["Force_(-D,-D)","Force_(0,-D)","Force_(D,-D)","Force_(-D,0)","Force_(0,0)","Force_(D,0)","Force_(-D,D)","Force_(0,D)","Force_(D,D)"];
% Figure_i_name = PilePosition;
% for i = 1:PileNum
%     fn(i)=figure(i);
%     saveas(figure(i), Figure_i_name(i), 'png');
% end

%%

XLabel = 'y (Pile displacement, m)';
YLabel = 'p (Soil resistance, kN/m)';
Grid = 'off';
Font = 'Arial';
FontSize = 7;
FigSize = [2,2,5.5,5]; % ���� ȭ��� x��ǥ, y��ǥ, ����, ���� ũ��
PlotXLim=[0, 0.10]; 
PlotYLim=[0 150];
FileType = '.emf'

pydepth_str = num2str(pydepth);
pydepth_str = strcat('_at_', pydepth_str, 'm');

% API sand pyĿ�� ����
[y_sand, p, k_initial1] = API_sand_v3(Water, D, gamma, phi, abs(pydepth), 0, PlotXLim(1), PlotXLim(2));
% Water: 0�̸� water table ��, 1�̸� water table �Ʒ�
% D: ���� ���� (m)
% gamma: �����߷� (kN/m^3)
% phi_deg: ���θ����� (degree)
% Z: py ������ ���� (m)
% cycllic: 0�̸� static loading, 1�̸� cyclic loading
% x��(p-y��� y��) ��������

PileCol = ["-k","-b","-r"];
PileRow = ["s","^","o"];

PyVarName_int = strcat('Py_SinglePile_out_int', pydepth_str, '.mat');
PyVarName_fit = strcat('Py_SinglePile_out_fit', pydepth_str, '.mat');

% interplotation���� �׸� ���������� py �
figure(21)
% plot(y_sand, p); % API �
load(PyVarName_int); % ���� pile �ؼ�������� ���� p-y��� �ҷ���. ���� ���� �ص־� �ҷ��� �� ����.
hold on
plot(Py_SinglePile_out_int(:,1),Py_SinglePile_out_int(:,2), "--k");
plot(zint_out(:,1), qint_out(:,1),strcat(PileCol(1,1),PileRow(1,1)));
plot(zint_out(:,2), qint_out(:,2),strcat(PileCol(1,2),PileRow(1,1)));
plot(zint_out(:,3), qint_out(:,3),strcat(PileCol(1,3),PileRow(1,1)));
plot(zint_out(:,4), qint_out(:,4),strcat(PileCol(1,1),PileRow(1,2)));
plot(zint_out(:,5), qint_out(:,5),strcat(PileCol(1,2),PileRow(1,2)));
plot(zint_out(:,6), qint_out(:,6),strcat(PileCol(1,3),PileRow(1,2)));
plot(zint_out(:,7), qint_out(:,7),strcat(PileCol(1,1),PileRow(1,3)));
plot(zint_out(:,8), qint_out(:,8),strcat(PileCol(1,2),PileRow(1,3)));
plot(zint_out(:,9), qint_out(:,9),strcat(PileCol(1,3),PileRow(1,3)));
hold off
% legend({'API','Single Pile','(-D,-D)','(0,-D)','(D,-D)','(-D,0)','(0,0)','(D,0)','(-D,D)','(0,D)','(D,D)'},'NumColumns', 1)

% �༭��
ax = gca;
ax.XLabel.String = XLabel;
ax.YLabel.String = YLabel;
ax.XLim=PlotXLim;
% ax.XTick=[0:0.1:0.6];
ax.XGrid = Grid;
ax.YLim = PlotYLim;
% ax.YTick = [0:5:25];
ax.YGrid = Grid;
ax.FontName = Font;
% ax.FontUnits = 'normalized'
ax.FontSize = FontSize;

% figure ���� ����
Figure_21_name = strcat('py � �� API vs Plaxis - spline ���', pydepth_str, FileType);
% title(Figure_21_name)
% ũ�� ����
set(gcf,'units','centimeters','position', FigSize);
% figure ����
saveas(gcf, Figure_21_name)


% polyfit���� �׸� ���������� py �
figure(22)
% plot(y_sand, p); % API �
load(PyVarName_fit); % ���� pile �ؼ�������� ���� p-y��� �ҷ���. ���� ���� �ص־� �ҷ��� �� ����.
hold on
plot(Py_SinglePile_out_fit(:,1),Py_SinglePile_out_fit(:,2));
plot(zf_out(:,1), qf_out(:,1),strcat(PileCol(1,1),PileRow(1,1)));
plot(zf_out(:,2), qf_out(:,2),strcat(PileCol(1,2),PileRow(1,1)));
plot(zf_out(:,3), qf_out(:,3),strcat(PileCol(1,3),PileRow(1,1)));
plot(zf_out(:,4), qf_out(:,4),strcat(PileCol(1,1),PileRow(1,2)));
plot(zf_out(:,5), qf_out(:,5),strcat(PileCol(1,2),PileRow(1,2)));
plot(zf_out(:,6), qf_out(:,6),strcat(PileCol(1,3),PileRow(1,2)));
plot(zf_out(:,7), qf_out(:,7),strcat(PileCol(1,1),PileRow(1,3)));
plot(zf_out(:,8), qf_out(:,8),strcat(PileCol(1,2),PileRow(1,3)));
plot(zf_out(:,9), qf_out(:,9),strcat(PileCol(1,3),PileRow(1,3)));
hold off
legend({'API','Single Pile','(-D,-D)','(0,-D)','(D,-D)','(-D,0)','(0,0)','(D,0)','(-D,D)','(0,D)','(D,D)'},'NumColumns', 1)

% �༭��
ax = gca;
ax.XLabel.String = XLabel;
ax.YLabel.String = YLabel;
ax.XLim=PlotXLim;
% ax.XTick=[0:0.1:0.6];
ax.XGrid = Grid;
ax.YLim = PlotYLim;
% ax.YTick = [0:5:25];
ax.YGrid = Grid;
ax.FontName = Font;
% ax.FontUnits = 'normalized'
ax.FontSize = FontSize;

% figure ���� ����
Figure_22_name = strcat('py � �� API vs Plaxis - ���׽� fitting ���', pydepth_str, FileType);
% title(Figure_22_name)
% ũ�� ����
set(gcf,'units','centimeters','position', FigSize);
% figure ����
saveas(gcf, Figure_22_name)

%% ������ ������

% 1. py curve of piles from interplotation
Py_int(:, 1:3:PileNum*3) = qint_out(:, 1:PileNum); % q data
Py_int(:, 2:3:PileNum*3) = zint_out(:, 1:PileNum); % y data
Filename_int = strcat('Py_GroupPile_out_int', pydepth_str, '.txt');
save(Filename_int, 'Py_int', '-ascii', '-double')

% 2. py curve of piles from polyfit
Py_fit(:, 1:2:PileNum*2) = qf_out(:, 1:PileNum); % q data
Py_fit(:, 2:2:PileNum*2) = zf_out(:, 1:PileNum); % y data
Filename_fit = strcat('Py_GroupPile_out_fit', pydepth_str, '.txt');
save(Filename_fit, 'Py_fit', '-ascii', '-double')

% 3. p-multiplier 
% p-multiplier�� ������ϴ� ���� py data�� Py_int, Py_fit, Py_SinglePile �����Ͽ� ���
% pm ����
pmx = [0.01:0.01:0.1];
Py_single_pm_int = interp1(Py_SinglePile_out_int(:,1), Py_SinglePile_out_int(:,2), pmx);
Py_single_pm_fit = interp1(Py_SinglePile_out_fit(:,1), Py_SinglePile_out_fit(:,2), pmx);
for i = 1:PileNum
    Py_group_pm_int(:,i) = interp1(zint_out(:,i), qint_out(:,i), pmx);
    Py_group_pm_fit(:,i) = interp1(zf_out(:,i), qf_out(:,i), pmx);
    pm_int(:,i) = Py_group_pm_int(:,i)./Py_single_pm_int';
    pm_fit(:,i) = Py_group_pm_fit(:,i)./Py_single_pm_fit';
end
Filename_pm_int = strcat('p-multiplier_int', pydepth_str, '.txt');
Filename_pm_fit = strcat('p-multiplier_fit', pydepth_str, '.txt');
save(Filename_pm_int, 'pm_int', '-ascii', '-double');
save(Filename_pm_fit, 'pm_fit', '-ascii', '-double');
