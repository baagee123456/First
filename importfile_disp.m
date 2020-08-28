function Untitled = importfile_disp(filename, dataLines)
%IMPORTFILE1 텍스트 파일에서 데이터 가져오기
%  UNTITLED = IMPORTFILE1(FILENAME)은 디폴트 선택 사항에 따라 텍스트 파일 FILENAME에서
%  데이터를 읽습니다.  데이터를 테이블로 반환합니다.
%
%  UNTITLED = IMPORTFILE1(FILE, DATALINES)는 텍스트 파일 FILENAME의 데이터를 지정된 행
%  간격으로 읽습니다. DATALINES를 양의 정수 스칼라로 지정하거나 양의 정수 스칼라로 구성된 Nx2 배열(인접하지 않은
%  행 간격인 경우)로 지정하십시오.
%
%  예:
%  Untitled = importfile1("C:\Users\Soil514-C\Desktop\PLAXIS 3D 해석결과\Pile Goup_Emb_inter(x)_1x1_Sand_1.p3dat\Report\3.1.1.1.1.1 Calculation results_ Embedded beam_ Phase_0.1m [Phase_2] (2_94)_ Table of total displacements.TXT", [1, Inf]);
%
%  READTABLE도 참조하십시오.
%
% MATLAB에서 2020-04-21 16:19:07에 자동 생성됨

%% 입력 처리

% dataLines를 지정하지 않는 경우 디폴트 값을 정의하십시오.
if nargin < 2
    dataLines = [1, Inf];
end

%% 가져오기 옵션을 설정하고 데이터 가져오기
opts = delimitedTextImportOptions("NumVariables", 10);

% 범위 및 구분 기호 지정
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% 열 이름과 유형 지정
opts.VariableNames = ["Structuralelement", "Node", "Localnumber", "Xm", "Ym", "Zm", "u_xm", "u_ym", "u_zm", "um"];
opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% 파일 수준 속성 지정
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 변수 속성 지정
opts = setvaropts(opts, "Structuralelement", "EmptyFieldRule", "auto");

% 데이터 가져오기
Untitled = readtable(filename, opts);

end