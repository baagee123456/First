function Untitled = importfile_disp(filename, dataLines)
%IMPORTFILE1 �ؽ�Ʈ ���Ͽ��� ������ ��������
%  UNTITLED = IMPORTFILE1(FILENAME)�� ����Ʈ ���� ���׿� ���� �ؽ�Ʈ ���� FILENAME����
%  �����͸� �н��ϴ�.  �����͸� ���̺�� ��ȯ�մϴ�.
%
%  UNTITLED = IMPORTFILE1(FILE, DATALINES)�� �ؽ�Ʈ ���� FILENAME�� �����͸� ������ ��
%  �������� �н��ϴ�. DATALINES�� ���� ���� ��Į��� �����ϰų� ���� ���� ��Į��� ������ Nx2 �迭(�������� ����
%  �� ������ ���)�� �����Ͻʽÿ�.
%
%  ��:
%  Untitled = importfile1("C:\Users\Soil514-C\Desktop\PLAXIS 3D �ؼ����\Pile Goup_Emb_inter(x)_1x1_Sand_1.p3dat\Report\3.1.1.1.1.1 Calculation results_ Embedded beam_ Phase_0.1m [Phase_2] (2_94)_ Table of total displacements.TXT", [1, Inf]);
%
%  READTABLE�� �����Ͻʽÿ�.
%
% MATLAB���� 2020-04-21 16:19:07�� �ڵ� ������

%% �Է� ó��

% dataLines�� �������� �ʴ� ��� ����Ʈ ���� �����Ͻʽÿ�.
if nargin < 2
    dataLines = [1, Inf];
end

%% �������� �ɼ��� �����ϰ� ������ ��������
opts = delimitedTextImportOptions("NumVariables", 10);

% ���� �� ���� ��ȣ ����
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% �� �̸��� ���� ����
opts.VariableNames = ["Structuralelement", "Node", "Localnumber", "Xm", "Ym", "Zm", "u_xm", "u_ym", "u_zm", "um"];
opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% ���� ���� �Ӽ� ����
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% ���� �Ӽ� ����
opts = setvaropts(opts, "Structuralelement", "EmptyFieldRule", "auto");

% ������ ��������
Untitled = readtable(filename, opts);

end