%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2020-05-27
% q_from_M_v7_VolumePile
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

% Soil parameter (py 곡선용)
Water =1;
gamma = 18-10; % 유효 단위중량
phi = 29;
pydepth = -1; % py curve 얻을 깊이

% Pile parameter 
PileLength = 10; % 아래 for문 안에도 Pile_L이라는 변수가 있음.
D = 0.5; % Pile diameter
Ansys = 10; % 해석 포인트 개수
PileNum = 9; % 무리말뚝 개수

tol1 = 0.00001; % py curve 얻을 깊이의 z index 찾때의 tolerance
RigidityFactor = 10^6; % Volume Pile 중간에 삽입된 Beam요소의 감소된 강성을 고려

% Plaxis data 불러오기
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

% zcol = [2, 169]; % 무리말뚝 중 해당 말뚝의 column 좌표 (매뉴얼 제어)
% 변수 공간 생성(속도 빠르게 하기 위함)
F_input_diff = zeros(Ansys, PileNum);
F_input_fit = zeros(Ansys, PileNum);
F_input_intp = zeros(Ansys, PileNum);
qf_out = zeros(Ansys+1, PileNum);
zf_out = zeros(Ansys+1, PileNum);
qint_out = zeros(Ansys+1, PileNum);
zint_out = zeros(Ansys+1, PileNum);

SampleData = importfile_force(FileName_force(1)); % 해당 말뚝의 column좌표 찾기위해 해석데이터 중 하나를 샘플로 불러옴
zz = SampleData(2:end,6); % z 좌표를 불러옴
zz = table2array(zz); % 불러온 값을 double 형으로 바꿈
[zzsize_m, zzsize_n] = size(zz); % zz의 사이즈 반환
DataSize = zzsize_m/PileNum; % 하나의 pile 데이터에 대한 data column 사이즈 반환

% [embedded beam 전용] ---------------------------------------------------
%  무리말뚝 중 해당 말뚝의 column 좌표 찾기 (나중에 이거 활용해서 추가 코딩 필요)
% SampleData = importfile_force(FileName_force(1)); % 해당 말뚝의 column좌표 찾기위해 해석데이터 중 하나를 샘플로 불러옴
% zz = SampleData(1:end,6); % z 좌표를 불러옴
% zz = table2array(zz); % 불러온 값을 double 형으로 바꿈
% zcol(:,1)=find(zz<=0.00000001 & zz>-0.00000001, PileNum); % z 좌표가 0인 곳의 column index를 불러옴
% zcol(:,2)=find(zz==-PileLength, PileNum); % z 좌표가 10인 곳의 column index를 불러옴
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

% 변수 공간 생성(속도 빠르게 하기 위함)
zarray = zeros(DataSize, PileNum);
Qarray = zeros(DataSize, PileNum);
Marray = zeros(DataSize, PileNum);
Dispxarray = zeros(DataSize, PileNum);

for i = 1:Ansys
    
    % 데이터 불러오기
    Data_Force = importfile_force(FileName_force(i));
    Data_Disp = importfile_disp(FileName_disp(i));
        
    for j = 1:PileNum
        
        Data_Force_PileNum = Data_Force(zcol(j,1):zcol(j,2),1:end); % 무리말뚝 중 j번째 말뚝에 대한 Force 데이터
        Data_Force_PileNum = sortrows(Data_Force_PileNum, 6, 'descend'); % 데이터를 깊이순으로 정렬
        z = Data_Force_PileNum(:,6); % Data_Force_PileNum에서 z값만 불러옴
        zarray(1:size(table2array(z)),j) = table2array(z); % 무리말뚝 j에 대해 j열에 z값 저장
        Pile_L(:,j) = min(zarray(:,j)); % Pile의 길이 반환 (무리말뚝 j에서 가장 깊은 z값)
        Q = Data_Force_PileNum(:,10); % Data_Force_PileNum에서 Q값만 불러옴
        Qarray(1:size(table2array(Q)),j) = table2array(Q)*RigidityFactor; % 무리말뚝 j에 대해 j열에 Q값 저장
        M = Data_Force_PileNum(:,19); % Data_Force_PileNum에서 M값만 불러옴
        Marray(1:size(table2array(M)),j) = table2array(M)*RigidityFactor; % 무리말뚝 j에 대해 j열에 M값 저장
        
        Data_Disp_PileNum = Data_Disp(zcol(j,1):zcol(j,2),1:end); % 무리말뚝 중 j번째 말뚝에 대한 Disp 데이터
        Data_Disp_PileNum = sortrows(Data_Disp_PileNum, 6, 'descend'); % 데이터를 깊이순으로 정렬
        Dispx =  Data_Disp_PileNum(:, 7); % Data_Disp_PileNum에서 Dispx값만 불러옴
        Dispxarray(1:size(table2array(Dispx)),j) = table2array(Dispx); % 무리말뚝 j에 대해 j열에 dispx값 저장
        
        
        %%% 방법1. 다항식으로 M를 피팅하지 않고 수치미분(차분)하여 Q와 q구함
        dz = diff(zarray(:,j)); % m
        dM = diff(Marray(:,j)); % kN-m
        Q_diff = dM./dz; % kN
        dQ_diff = diff(Q_diff); % kN
        q_diff = dQ_diff./dz(2:end); % kn/m
        
        
        %%% 방법2. 다항식으로 M선도를 피팅하고 이를 미분하여 Q와 q구함
        polyfit_M = polyfitB(zarray(:,j),Marray(:,j), 6, 0); % z에 따른 M를 다항식 피팅
        % polyfitB는 특정 y절편을 가지는 fitting은 사용자 함수
        % polyfitB(x, y, 차수, y절편)
        % https://kr.mathworks.com/matlabcentral/fileexchange/35401-polyfitzero 참고
        % 두점을 지나는 polyfit은
        % https://kr.mathworks.com/matlabcentral/fileexchange/54207-polyfix-x-y-n-xfix-yfix-xder-dydx
        
        % 피팅된 Q 다항식 plot을 위한 변수생성
        delta = 0.01;
        zf = 0:-delta:Pile_L;
        Mf = polyval(polyfit_M, zf);
        
        % Q 다항식을 미분하여 Q (sheer force, kN), q (soil resistance, kN/m) 구함
        polyfit_Q = polyder(polyfit_M);
        polyfit_q = polyder(polyfit_Q);
        Qf = polyval(polyfit_Q, zf);
        qf = polyval(polyfit_q, zf);
        
        
        %%% 방법3. M data를 interplotation하고, 이를 수치미분하여 Q와 q구함
        SampRate = 1; % PileLength에서의 index 찾을 때 sampling rate (m)
        factor1 = 1 / SampRate; % PileLength에서 sampling된 데이터를 각 행(정수)로 저장하기 위한 작업
        tol2 = 0.0001; % PileLength에서의 index를 찾을 때 오차 tolerance
        zarray_intp = zeros(PileLength * factor1 + 1, 1); % 첫째 z값 0으로 함.
        Marray_intp = zeros(PileLength * factor1 + 1, 1); % 첫째 M값 0으로 함.
        for nn = SampRate:SampRate:PileLength
            zcol2(nn*factor1,:)=find(zarray(:,j) >= -nn-tol2 & zarray(:,j) <= -nn+tol2, 1); % z = nn (m) PileLength에서의 index를 찾음
            zarray_intp(1+nn*factor1,:) = zarray(zcol2(nn*factor1),j); % 0 이후 z
            Marray_intp(1+nn*factor1,:) =Marray(zcol2(nn*factor1),j); % 0 이후의 M
        end
        % 앞서 nn (m)에서 sampling한 zarray_intp, Marray_intp를 zintp 지점에서 보간
        zintp = [-PileLength:0.0001:0]; % interplotation할 z좌표
        Mintp = interp1(zarray_intp, Marray_intp, zintp, 'spline'); % zintp에서 interplotation 실시
        % M을 수치미분
        dz_intp = diff(zintp); % m
        dM_intp = diff(Mintp); % kN-m
        Q_diff_intp = dM_intp./dz_intp; % kN
        dQ_diff_intp = diff(Q_diff_intp); % kN
        q_diff_intp = dQ_diff_intp./dz_intp(2:end); % kn/m
        
        
%         % plot
%         figure(j) % figure 1-j는 각 무리말뚝의 위치
%         % M plot
%         subplot(3,Ansys,i);  % 1-3행은 각각 Bending Moment, Shear forece, soil resistance; 1-i열은 하중 단계
%         plot(zarray(:,j), Marray(:,j), 'o', zf, Mf, zintp, Mintp, 'k');
%         xlabel('z (m)')
%         ylabel('Moment (kN-m)')
%         legend('Plaxis', '다항식 fitting','Spline')
%         grid on
%         % Q plot
%         subplot(3,Ansys,i+Ansys)
%         plot(zarray(2:end,j), Q_diff, 'o', zf, Qf, zintp(2:end), Q_diff_intp,'k');
%         xlabel('z (m)')
%         ylabel('Shear force (kN)')
%         grid on
%         % q plot
%         subplot(3,Ansys,i+Ansys*2)
%         plot(zarray(3:end,j), q_diff, 'o', zf, qf, zintp(3:end), q_diff_intp,'k');  % 수치미분값과 pit함수 미분값
%         xlabel('z (m)')
%         ylabel('Soil resistance (kN/m)')
%         ylim([-200*i 200*i])
%         grid on
%         f=figure(j);
%         f.Position = [30 80  1800 800];
        
        % % figure 저장
        % pydepth_str = num2str(pydepth);
        % pydepth_str = strcat('_at_', pydepth_str, 'm');
        % PilePosition = ["Force_(D,D)","Force_(0,D)","Force_(-D,D)","Force_(D,0)","Force_(0,0)","Force_(-D,0)","Force_(D,-D)","Force_(0,-D)","Force_(-D,-D)"];
        % Figure_i_name = strcat(PilePosition, pydepth_str);
        % saveas(gcf, Figure_i_name(j), 'png')
                
        % 수치 미적분값이 적절한지 확인
        F_input_diff(i,j)=Qarray(1,j)-Qarray(end,j); % 파일 head에 가해진 하중을 역계산하여 확인 (Plaxis 결과)
        F_input_fit(i,j)=Qf(1)-Qf(end); % 파일 head에 가해진 하중을 역계산하여 확인 (다항식 피팅 방법)
        F_input_intp(i,j)=Q_diff_intp(1)-Q_diff_intp(end); % 파일 head에 가해진 하중을 역계산하여 확인 (다항식 피팅 방법)
        
        qf_out(1,j)=0; % (0,0)인 점을 플롯하기 위해 원점좌표 생성
        zf_out(1,j)=0; % (0,0)인 점을 플롯하기 위해 원점좌표 생성
        qf_out(i+1,j) = abs(polyval(polyfit_q,pydepth)); % pydepth 값에서의 soil resistance를 구함
        zindex = find(zarray(:,j)>pydepth-tol1 & zarray(:,j)<pydepth+tol1, 1); % pydepth값에 가장 가까운 zindex를 찾음
        zf_out(i+1,j) = Dispxarray(min(zindex),j); % 찾은 index에 해당하는 pile disp 값을 반환
        
        qint_out(1,j)=0; % (0,0)인 점을 플롯하기 위해 원점좌표 생성
        zint_out(1,j)=0; % (0,0)인 점을 플롯하기 위해 원점좌표 생성
        z3 = zintp(3:end); % q_diff_intp 배열에 대응하는 z값
        pydepth_index1 = find(z3==pydepth,1); % z3 값 중 pydepth에 해당하는 index 구함
        qint_out(i+1,j) = abs(q_diff_intp(pydepth_index1)); % pydepth 값에서의 soil resistance를 구함
        pydepth_index2 = find(zarray(:,j)>pydepth-tol1 & zarray(:,j)<pydepth+tol1, 1); % pydepth값에 가장 가까운 zindex를 찾음
        zint_out(i+1,j) = Dispxarray(min(pydepth_index2),j);
    end
end

% % figure 저장
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
FigSize = [2,2,5.5,5]; % 각각 화면상 x좌표, y좌표, 가로, 세로 크기
PlotXLim=[0, 0.10]; 
PlotYLim=[0 150];
FileType = '.emf'

pydepth_str = num2str(pydepth);
pydepth_str = strcat('_at_', pydepth_str, 'm');

% API sand py커브 생성
[y_sand, p, k_initial1] = API_sand_v3(Water, D, gamma, phi, abs(pydepth), 0, PlotXLim(1), PlotXLim(2));
% Water: 0이면 water table 위, 1이면 water table 아래
% D: 파일 직경 (m)
% gamma: 단위중량 (kN/m^3)
% phi_deg: 내부마찰각 (degree)
% Z: py 산출할 깊이 (m)
% cycllic: 0이면 static loading, 1이면 cyclic loading
% x값(p-y곡선의 y값) 범위설정

PileCol = ["-k","-b","-r"];
PileRow = ["s","^","o"];

PyVarName_int = strcat('Py_SinglePile_out_int', pydepth_str, '.mat');
PyVarName_fit = strcat('Py_SinglePile_out_fit', pydepth_str, '.mat');

% interplotation으로 그린 무리말뚝의 py 곡선
figure(21)
% plot(y_sand, p); % API 곡선
load(PyVarName_int); % 단일 pile 해석결과에서 얻은 p-y곡선을 불러옴. 따로 저장 해둬야 불러올 수 있음.
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

% 축서식
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

% figure 제목 지정
Figure_21_name = strcat('py 곡선 비교 API vs Plaxis - spline 방법', pydepth_str, FileType);
% title(Figure_21_name)
% 크기 지정
set(gcf,'units','centimeters','position', FigSize);
% figure 저장
saveas(gcf, Figure_21_name)


% polyfit으로 그린 무리말뚝의 py 곡선
figure(22)
% plot(y_sand, p); % API 곡선
load(PyVarName_fit); % 단일 pile 해석결과에서 얻은 p-y곡선을 불러옴. 따로 저장 해둬야 불러올 수 있음.
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

% 축서식
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

% figure 제목 지정
Figure_22_name = strcat('py 곡선 비교 API vs Plaxis - 다항식 fitting 방법', pydepth_str, FileType);
% title(Figure_22_name)
% 크기 지정
set(gcf,'units','centimeters','position', FigSize);
% figure 저장
saveas(gcf, Figure_22_name)

%% 저장할 변수들

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
% p-multiplier를 얻고자하는 점의 py data를 Py_int, Py_fit, Py_SinglePile 보간하여 얻고
% pm 구함
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
