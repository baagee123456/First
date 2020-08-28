function ForcePhase1 = importfile_force(filename, dataLines)
%IMPORTFILE1 텍스트 파일에서 데이터 가져오기
%  FORCEPHASE1 = IMPORTFILE1(FILENAME)은 디폴트 선택 사항에 따라 텍스트 파일 FILENAME에서
%  데이터를 읽습니다.  데이터를 테이블로 반환합니다.
%
%  FORCEPHASE1 = IMPORTFILE1(FILE, DATALINES)는 텍스트 파일 FILENAME의 데이터를 지정된
%  행 간격으로 읽습니다. DATALINES를 양의 정수 스칼라로 지정하거나 양의 정수 스칼라로 구성된 Nx2 배열(인접하지
%  않은 행 간격인 경우)로 지정하십시오.
%
%  예:
%  ForcePhase1 = importfile1("C:\Users\Soil514-C\Desktop\PLAXIS 3D 해석결과\Pile Goup_Emb_inter(x)_1x1_Sand_1.p3dat\Report\Force_Phase1.TXT", [1, Inf]);
%
%  READTABLE도 참조하십시오.
%
% MATLAB에서 2020-04-21 16:47:25에 자동 생성됨

%% 입력 처리

% dataLines를 지정하지 않는 경우 디폴트 값을 정의하십시오.
if nargin < 2
    dataLines = [1, Inf];
end

%% 가져오기 옵션을 설정하고 데이터 가져오기
opts = delimitedTextImportOptions("NumVariables", 31);

% 범위 및 구분 기호 지정
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% 열 이름과 유형 지정
opts.VariableNames = ["Structuralelement", "Node", "Localnumber", "Xm", "Ym", "Zm", "NkN", "N_minkN", "N_maxkN", "Q_12kN", "Q_12minkN", "Q_12maxkN", "Q_13kN", "Q_13minkN", "Q_13maxkN", "M_2kNm", "M_2minkNm", "M_2maxkNm", "M_3kNm", "M_3minkNm", "M_3maxkNm", "T_skinkNm", "T_skinminkNm", "T_skinmaxkNm", "T_latkNm", "T_latminkNm", "T_latmaxkNm", "T_lat2kNm", "T_lat2minkNm", "T_lat2maxkNm", "F_footkN"];
opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical"];

% 파일 수준 속성 지정
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 변수 속성 지정
opts = setvaropts(opts, ["Structuralelement", "F_footkN"], "EmptyFieldRule", "auto");

% 데이터 가져오기
ForcePhase1 = readtable(filename, opts);

end