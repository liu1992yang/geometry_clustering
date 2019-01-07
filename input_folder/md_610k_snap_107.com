%mem=64gb
%nproc=28       
%Chk=snap_107.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_107 

2     1 
  O    3.398653575  -4.271217188  -5.914485967
  C    3.559333873  -3.093897539  -6.703534950
  H    4.277462630  -3.300027069  -7.522191006
  H    4.027289176  -2.381488382  -5.989068614
  C    2.241431209  -2.546499225  -7.259368440
  H    2.295241699  -1.450880921  -7.474145722
  O    2.038837152  -3.135610575  -8.589205255
  C    0.877761419  -3.945455901  -8.623685952
  H    0.352282753  -3.708541272  -9.582187213
  C    0.994511663  -2.900074546  -6.420018996
  H    1.273040215  -3.427166845  -5.468122745
  C    0.093742023  -3.746750844  -7.329784353
  H   -0.887016244  -3.256527510  -7.519021174
  H   -0.168849570  -4.710696839  -6.832554344
  O    0.353481388  -1.636437314  -6.139136818
  N    1.367541018  -5.378926529  -8.737419634
  C    0.567197670  -6.527850610  -8.572420367
  C    1.429675778  -7.652008559  -8.722850306
  N    2.730707181  -7.171535273  -8.919567138
  C    2.685611575  -5.807710296  -8.933612320
  N   -0.786407520  -6.579759663  -8.368378906
  C   -1.298121413  -7.838892611  -8.243919133
  N   -0.536181488  -9.008832059  -8.439304932
  C    0.911508938  -9.003689825  -8.622032045
  N   -2.629664880  -7.971466193  -7.847347387
  H   -3.121191397  -7.072769649  -7.661257370
  H   -3.215557026  -8.654492431  -8.319450196
  O    1.483984913 -10.058729345  -8.691473902
  H    3.523593115  -5.123591849  -9.096475825
  H   -0.967491478  -9.938003839  -8.333574344
  P   -0.576491929  -1.672765913  -4.763054027
  O   -0.780690025  -0.028670970  -4.686438377
  C   -0.028113128   0.688557581  -3.680420737
  H    0.650450712   1.362084607  -4.255010819
  H    0.579382716   0.008577137  -3.044126551
  C   -1.020278924   1.502351512  -2.850797808
  H   -0.647191949   1.641248497  -1.803770786
  O   -1.047163746   2.846140520  -3.433021308
  C   -2.366668485   3.400835950  -3.369222970
  H   -2.274133714   4.351089019  -2.783745334
  C   -2.500203779   1.031060104  -2.858773389
  H   -2.768109072   0.442376226  -3.770487812
  C   -3.298871670   2.339937724  -2.773820622
  H   -3.546449780   2.559011704  -1.706771243
  H   -4.284413689   2.275433103  -3.269837955
  O   -2.771136072   0.289684704  -1.667549443
  O    0.118183895  -2.377942670  -3.670860346
  O   -1.958962378  -2.117131153  -5.501297107
  N   -2.752837365   3.770028304  -4.766837848
  C   -2.729400826   3.012750079  -5.934665980
  C   -3.262672781   3.844415957  -6.976342979
  N   -3.594765668   5.102102060  -6.451962207
  C   -3.291242692   5.058068987  -5.157328702
  N   -2.324194236   1.709952273  -6.208772571
  C   -2.432818365   1.284925991  -7.541732770
  N   -2.930788380   2.013211886  -8.531898038
  C   -3.374994747   3.333168203  -8.308117517
  H   -2.083448410   0.249611350  -7.762202672
  N   -3.881792076   4.021952246  -9.345327002
  H   -3.936083170   3.617823550 -10.279470430
  H   -4.208116967   4.982235094  -9.228927233
  H   -3.416460571   5.864905549  -4.438121215
  P   -2.549936880  -1.354334905  -1.812007529
  O   -1.332818208  -1.427637324  -0.740684930
  C   -0.510975937  -2.631160106  -0.762811495
  H   -0.105531450  -2.816021421  -1.787491008
  H    0.323001747  -2.366052996  -0.077283569
  C   -1.396519336  -3.744188280  -0.221955138
  H   -1.661213967  -3.598368864   0.854117870
  O   -2.660200608  -3.535453870  -0.942484351
  C   -3.147592618  -4.780341671  -1.527206263
  H   -4.214494892  -4.815208291  -1.188111317
  C   -0.968648310  -5.200362373  -0.504721812
  H   -0.142240830  -5.301211139  -1.250497647
  C   -2.264032991  -5.893192646  -0.965770288
  H   -2.054074774  -6.717541758  -1.676331439
  H   -2.744063636  -6.390153053  -0.087927758
  O   -0.648802428  -5.828181441   0.755065941
  O   -2.286065715  -1.710501903  -3.248622604
  O   -4.029201793  -1.779673707  -1.287384980
  N   -3.121537268  -4.649696895  -3.009235750
  C   -2.031750058  -5.148071286  -3.797576238
  N   -2.312114986  -5.352116174  -5.168520594
  C   -3.535785081  -4.971665223  -5.815825587
  C   -4.428807387  -4.144049008  -4.984361597
  C   -4.233910413  -4.042481544  -3.646604240
  O   -0.950797104  -5.431089027  -3.317889575
  H   -1.625517038  -5.918387771  -5.707308974
  O   -3.715632858  -5.365946222  -6.949508338
  C   -5.562921897  -3.491586507  -5.691827781
  H   -6.310704270  -4.241102561  -6.010581443
  H   -6.090586591  -2.746914188  -5.081374986
  H   -5.228427455  -2.985793546  -6.612711698
  H   -4.919085909  -3.484688975  -2.987818230
  P    0.924470523  -5.826170970   1.210969111
  O    1.588968114  -6.372192574  -0.192826858
  C    3.031634093  -6.559499007  -0.184533309
  H    3.504138026  -6.223204988   0.759524052
  H    3.172877250  -7.660862073  -0.282922154
  C    3.619842234  -5.828614183  -1.392934162
  H    4.729013287  -5.733077097  -1.269390123
  O    3.489898563  -6.695792259  -2.556065261
  C    2.708146571  -6.078707842  -3.584481470
  H    3.366667483  -6.082250576  -4.490728833
  C    2.969563069  -4.465825001  -1.758774381
  H    2.312383126  -4.059135513  -0.956811439
  C    2.263733107  -4.697203831  -3.099497812
  H    1.157499385  -4.605231009  -3.015642270
  H    2.527709845  -3.908395273  -3.840404903
  O    3.977156945  -3.467562965  -1.845372772
  O    1.267662805  -6.441993921   2.481723782
  O    1.291428391  -4.212031595   1.058476735
  N    1.541629407  -6.991417934  -3.866184865
  C    0.932260509  -6.934929944  -5.186230699
  N   -0.163060147  -7.730112690  -5.452248115
  C   -0.681495214  -8.530327498  -4.463531435
  C   -0.108092578  -8.581011504  -3.153557194
  C    0.986834607  -7.792238256  -2.882003587
  O    1.413666448  -6.170673484  -6.022387271
  N   -1.825809560  -9.216834702  -4.797888091
  H   -2.186628870  -9.953465191  -4.211634096
  H   -2.149131109  -9.189529336  -5.755505674
  H    2.799386633  -4.937532522  -6.345281133
  H    1.462532120  -7.768724243  -1.884740712
  H   -0.519738127  -9.232291494  -2.386239777
  H    4.546114632  -3.588634041  -2.637729205
  H   -2.718562903  -2.399248029  -4.869797353
  H   -4.117688013  -2.095960777  -0.335657084
  H    1.728684602  -3.761164755   1.828819993
  H    3.548512945  -7.765503512  -9.033471518
  H   -1.854379937   1.116257753  -5.486047617

