%mem=64gb
%nproc=28       
%Chk=snap_103.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_103 

2     1 
  O    2.866142919  -4.757166253  -7.421775244
  C    2.553966733  -3.519419509  -8.066740199
  H    3.123075415  -3.603027462  -9.023293297
  H    2.962475930  -2.685213368  -7.463632828
  C    1.068166101  -3.300260446  -8.357865018
  H    0.912858427  -2.281896978  -8.799234314
  O    0.679892756  -4.219484263  -9.435923957
  C   -0.541460377  -4.879966094  -9.132591107
  H   -1.126587526  -4.928121507 -10.081845313
  C    0.050966857  -3.532240243  -7.210542533
  H    0.448992112  -4.179327849  -6.393823001
  C   -1.160878998  -4.152215494  -7.934652710
  H   -1.867272093  -3.361874460  -8.269067045
  H   -1.775989077  -4.804188746  -7.279236620
  O   -0.228898012  -2.216369634  -6.706525535
  N   -0.194937199  -6.296450857  -8.729186821
  C   -1.115791146  -7.363680968  -8.618480384
  C   -0.512925241  -8.330979689  -7.763288584
  N    0.765355797  -7.863216846  -7.411588590
  C    0.938799831  -6.642559905  -7.969762366
  N   -2.330472153  -7.506098596  -9.230759247
  C   -3.005897447  -8.650263963  -8.890700551
  N   -2.510783135  -9.621889366  -7.984315198
  C   -1.230528209  -9.491028418  -7.306285661
  N   -4.252066064  -8.840050416  -9.445121867
  H   -4.576971971  -8.181696951 -10.151347429
  H   -4.727191704  -9.730169043  -9.410530799
  O   -0.927498561 -10.316214470  -6.474585611
  H    1.865054260  -5.963954349  -7.891234441
  H   -3.093570435 -10.421446467  -7.702812029
  P   -1.031643486  -2.106553684  -5.256502801
  O   -1.921114209  -0.761213849  -5.637419564
  C   -1.373003377   0.496841150  -5.220427025
  H   -1.797358692   1.198285667  -5.970480081
  H   -0.268790886   0.528990748  -5.283602931
  C   -1.880787458   0.811078191  -3.800803068
  H   -1.212760210   0.416371082  -2.995213254
  O   -1.777011025   2.255642604  -3.603964041
  C   -3.066786882   2.872227851  -3.569507176
  H   -3.019719802   3.602509994  -2.720004592
  C   -3.375926704   0.436700235  -3.626293791
  H   -3.767607401  -0.178935692  -4.476039400
  C   -4.120418487   1.774879500  -3.457821018
  H   -4.607649546   1.784825598  -2.448222695
  H   -4.957924070   1.862063855  -4.173124164
  O   -3.632454984  -0.252720743  -2.392436389
  O   -0.074065642  -1.946733903  -4.136994642
  O   -2.045310535  -3.354515427  -5.480097469
  N   -3.167796220   3.657597359  -4.851713886
  C   -2.091779259   4.117775960  -5.603124133
  C   -2.634469021   4.870202832  -6.694608774
  N   -4.033230291   4.874741270  -6.605469201
  C   -4.343876028   4.165276795  -5.519609063
  N   -0.715604862   3.957440416  -5.440761584
  C    0.097699618   4.599837308  -6.390882394
  N   -0.357397518   5.303084281  -7.415201369
  C   -1.741616473   5.484622160  -7.631333368
  H    1.198824508   4.495039857  -6.266918774
  N   -2.126999953   6.209746179  -8.691960909
  H   -1.446659769   6.632251815  -9.325287970
  H   -3.115571703   6.373386196  -8.887701902
  H   -5.345764520   3.983877859  -5.137577167
  P   -2.936388345  -1.741702231  -2.186923935
  O   -1.750423365  -1.204154001  -1.201695502
  C   -0.589989626  -2.072199108  -1.096756992
  H   -0.228192793  -2.356659046  -2.116560254
  H    0.161944025  -1.425009575  -0.596253846
  C   -1.036783723  -3.252573535  -0.242508153
  H   -1.265370052  -2.966133731   0.811088157
  O   -2.332704479  -3.607565709  -0.844687283
  C   -2.423549628  -5.050151060  -1.074810115
  H   -3.405230766  -5.316937685  -0.610652986
  C   -0.174732394  -4.530106173  -0.305416657
  H    0.534881775  -4.514175986  -1.179910375
  C   -1.204454584  -5.670739265  -0.394386682
  H   -0.792964292  -6.558918135  -0.916565569
  H   -1.454027497  -6.033849247   0.628334167
  O    0.515106765  -4.624683574   0.949931410
  O   -2.519945565  -2.333032484  -3.497449931
  O   -4.193916996  -2.443695869  -1.442376772
  N   -2.492397901  -5.294301834  -2.539202706
  C   -1.309073306  -5.509960948  -3.321081530
  N   -1.497476763  -6.066064978  -4.601724302
  C   -2.773365690  -6.300224526  -5.213909462
  C   -3.929589950  -5.866645291  -4.405810941
  C   -3.772014298  -5.401240912  -3.143332113
  O   -0.194403066  -5.253506412  -2.901216378
  H   -0.628226105  -6.322656153  -5.115520988
  O   -2.774920183  -6.799853305  -6.318630065
  C   -5.259401854  -5.979071000  -5.063965729
  H   -6.101122525  -5.778792677  -4.386665888
  H   -5.342633511  -5.280780281  -5.914204145
  H   -5.418294919  -6.991749356  -5.475127877
  H   -4.620978096  -5.073328195  -2.524602741
  P    2.084091685  -5.153898090   0.832929805
  O    2.705352603  -3.707408222   0.357211067
  C    3.999671236  -3.686986813  -0.284408643
  H    4.499623736  -2.834993359   0.232301349
  H    4.584945882  -4.614560313  -0.113888907
  C    3.865578834  -3.403788765  -1.782919694
  H    4.631390995  -2.654107389  -2.105048283
  O    4.235509996  -4.643220413  -2.475689285
  C    3.436401585  -4.828601471  -3.652195301
  H    4.161168379  -4.853078187  -4.503166512
  C    2.473271529  -3.023649842  -2.329102384
  H    1.650815189  -3.342496619  -1.649642858
  C    2.388440358  -3.716966708  -3.696016149
  H    1.354974330  -4.094233029  -3.884274885
  H    2.581345661  -3.001038915  -4.517045022
  O    2.431649840  -1.602342246  -2.455298000
  O    2.390209163  -6.308489202  -0.015246734
  O    2.433248999  -5.278063057   2.408515429
  N    2.797694934  -6.194376257  -3.570339529
  C    1.863279221  -6.568412470  -4.601417041
  N    1.375747913  -7.850742233  -4.651928004
  C    1.667129049  -8.741938246  -3.639028231
  C    2.491696708  -8.340825288  -2.536862550
  C    3.040984399  -7.081234548  -2.520030930
  O    1.514968730  -5.729723256  -5.449935972
  N    1.114754861  -9.981669163  -3.776247970
  H    1.304473127 -10.726043888  -3.122249542
  H    0.523677167 -10.203129672  -4.579254653
  H    2.532340159  -4.785567819  -6.473816010
  H    3.692586198  -6.722393040  -1.698460180
  H    2.672639400  -9.015086156  -1.698139646
  H    1.674312110  -1.352475343  -3.053748514
  H   -2.666071334  -3.570622838  -4.681215727
  H   -4.123857989  -2.584544281  -0.445227120
  H    2.111872345  -4.557677682   3.026228922
  H    1.325521713  -8.278749145  -6.655772170
  H   -0.360815906   3.413279396  -4.637195056
