function ForcePhase1 = importfile_force(filename, dataLines)
%IMPORTFILE1 �ؽ�Ʈ ���Ͽ��� ������ ��������
%  FORCEPHASE1 = IMPORTFILE1(FILENAME)�� ����Ʈ ���� ���׿� ���� �ؽ�Ʈ ���� FILENAME����
%  �����͸� �н��ϴ�.  �����͸� ���̺�� ��ȯ�մϴ�.
%
%  FORCEPHASE1 = IMPORTFILE1(FILE, DATALINES)�� �ؽ�Ʈ ���� FILENAME�� �����͸� ������
%  �� �������� �н��ϴ�. DATALINES�� ���� ���� ��Į��� �����ϰų� ���� ���� ��Į��� ������ Nx2 �迭(��������
%  ���� �� ������ ���)�� �����Ͻʽÿ�.
%
%  ��:
%  ForcePhase1 = importfile1("C:\Users\Soil514-C\Desktop\PLAXIS 3D �ؼ����\Pile Goup_Emb_inter(x)_1x1_Sand_1.p3dat\Report\Force_Phase1.TXT", [1, Inf]);
%
%  READTABLE�� �����Ͻʽÿ�.
%
% MATLAB���� 2020-04-21 16:47:25�� �ڵ� ������

%% �Է� ó��

% dataLines�� �������� �ʴ� ��� ����Ʈ ���� �����Ͻʽÿ�.
if nargin < 2
    dataLines = [1, Inf];
end

%% �������� �ɼ��� �����ϰ� ������ ��������
opts = delimitedTextImportOptions("NumVariables", 31);

% ���� �� ���� ��ȣ ����
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% �� �̸��� ���� ����
opts.VariableNames = ["Structuralelement", "Node", "Localnumber", "Xm", "Ym", "Zm", "NkN", "N_minkN", "N_maxkN", "Q_12kN", "Q_12minkN", "Q_12maxkN", "Q_13kN", "Q_13minkN", "Q_13maxkN", "M_2kNm", "M_2minkNm", "M_2maxkNm", "M_3kNm", "M_3minkNm", "M_3maxkNm", "T_skinkNm", "T_skinminkNm", "T_skinmaxkNm", "T_latkNm", "T_latminkNm", "T_latmaxkNm", "T_lat2kNm", "T_lat2minkNm", "T_lat2maxkNm", "F_footkN"];
opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical"];

% ���� ���� �Ӽ� ����
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% ���� �Ӽ� ����
opts = setvaropts(opts, ["Structuralelement", "F_footkN"], "EmptyFieldRule", "auto");

% ������ ��������
ForcePhase1 = readtable(filename, opts);

end