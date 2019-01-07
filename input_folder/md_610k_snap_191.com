%mem=64gb
%nproc=28       
%Chk=snap_191.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_191 

2     1 
  O    2.709002240  -4.299642836  -4.782624075
  C    2.817180692  -3.156400861  -5.632775683
  H    3.811891136  -3.303582363  -6.109877151
  H    2.836730991  -2.246719246  -4.998825387
  C    1.720523768  -3.052466296  -6.698065770
  H    1.692952144  -2.031995010  -7.158073837
  O    2.071137650  -3.904889858  -7.830256592
  C    1.137923758  -4.961074518  -7.998363424
  H    0.943021492  -5.018383094  -9.101712471
  C    0.315889538  -3.491988093  -6.226577623
  H    0.296301524  -3.746052499  -5.135551049
  C   -0.071423365  -4.692542126  -7.103059685
  H   -0.974621371  -4.450237294  -7.707346766
  H   -0.374895569  -5.559151986  -6.475564277
  O   -0.559436105  -2.383121424  -6.548459540
  N    1.857456063  -6.234232542  -7.619637934
  C    2.333314175  -7.192593708  -8.536524553
  C    3.067258221  -8.157642212  -7.784411673
  N    3.040066255  -7.758940293  -6.433975269
  C    2.328819433  -6.610695083  -6.345136237
  N    2.130644909  -7.227892957  -9.888673780
  C    2.710568537  -8.292226029 -10.531992286
  N    3.463401780  -9.289518151  -9.861713831
  C    3.690998804  -9.283070098  -8.426695910
  N    2.537467335  -8.383962290 -11.887291811
  H    2.007003242  -7.661847104 -12.372964857
  H    2.951979744  -9.111166834 -12.451233332
  O    4.353794762 -10.189882419  -7.971323641
  H    2.141345207  -6.004882136  -5.421511608
  H    3.883444637 -10.073162044 -10.383537107
  P   -1.590369652  -1.959063377  -5.332949536
  O   -2.531030199  -0.822657371  -6.054812557
  C   -1.902004486   0.431690544  -6.395464155
  H   -2.713882675   0.947816544  -6.953698633
  H   -1.055809190   0.267417495  -7.093086967
  C   -1.488914536   1.222187920  -5.145094483
  H   -0.440611493   1.009996469  -4.808892350
  O   -1.450650357   2.627182436  -5.531156530
  C   -2.373948628   3.405350522  -4.770658967
  H   -1.806685359   4.330992764  -4.483214806
  C   -2.486119998   1.108234506  -3.962970420
  H   -3.374695189   0.481949481  -4.214219583
  C   -2.860360019   2.557811469  -3.602404309
  H   -2.307534607   2.833212020  -2.664159951
  H   -3.918144897   2.673264438  -3.314508049
  O   -1.758306465   0.608715421  -2.835426428
  O   -0.751919641  -1.513523115  -4.155314617
  O   -2.488230255  -3.289867924  -5.276296483
  N   -3.429524818   3.840865241  -5.747451252
  C   -4.798098272   3.609174343  -5.833725914
  C   -5.272663985   4.358240186  -6.964168733
  N   -4.196184688   5.004421353  -7.589527101
  C   -3.115718280   4.691768506  -6.879322628
  N   -5.686393375   2.862427094  -5.066927502
  C   -7.040920872   2.914013752  -5.433829030
  N   -7.523596396   3.611765997  -6.452034684
  C   -6.670024150   4.376815908  -7.271403532
  H   -7.743763266   2.317069986  -4.806873010
  N   -7.216096242   5.078834775  -8.279617183
  H   -8.220759215   5.070630277  -8.450022726
  H   -6.637911953   5.649143857  -8.895969758
  H   -2.090423095   4.999686804  -7.078328773
  P   -2.089275660  -0.958772102  -2.414012569
  O   -0.739715952  -1.395932118  -1.621211048
  C   -0.327925770  -2.769668245  -1.780399878
  H    0.061661082  -2.922630456  -2.807532816
  H    0.525324800  -2.832259431  -1.069436424
  C   -1.472938199  -3.706025717  -1.376655556
  H   -1.247171119  -4.231063174  -0.416611433
  O   -2.622418277  -2.841951657  -1.123727118
  C   -3.856090156  -3.656562905  -1.249768866
  H   -4.130244516  -3.917755295  -0.196326055
  C   -1.993843452  -4.708696575  -2.439599405
  H   -1.837585162  -4.399020880  -3.501611536
  C   -3.481244317  -4.874930539  -2.097176627
  H   -4.108240297  -4.999285142  -3.011769160
  H   -3.636033432  -5.827024205  -1.538129126
  O   -1.391827645  -6.006663059  -2.193280863
  O   -2.929173136  -1.657364197  -3.478119234
  O   -3.024576688  -0.545041694  -1.113632542
  N   -4.887627381  -2.725749176  -1.777272629
  C   -5.029504732  -2.510358501  -3.186361980
  N   -5.541126176  -1.249820862  -3.606692764
  C   -5.809841611  -0.171180781  -2.722091926
  C   -5.842929374  -0.526425405  -1.300800691
  C   -5.331844599  -1.711428331  -0.878684236
  O   -4.866154275  -3.401208825  -4.009974430
  H   -5.549933644  -1.087574788  -4.624686197
  O   -5.983206166   0.937379817  -3.219509067
  C   -6.415261562   0.479758884  -0.365968560
  H   -7.506060022   0.353188629  -0.261710309
  H   -5.982372342   0.421856509   0.643292783
  H   -6.242766486   1.508787516  -0.722554616
  H   -5.258275002  -1.964468256   0.187835597
  P    0.131427004  -6.151517156  -2.770355309
  O    0.944275343  -5.281741994  -1.618049243
  C    2.361920179  -5.504448884  -1.461627489
  H    2.858939527  -5.827374737  -2.403455997
  H    2.729816439  -4.485568285  -1.202811424
  C    2.572284638  -6.472168067  -0.294357857
  H    1.770435646  -6.330898847   0.473317880
  O    2.365774153  -7.855150947  -0.721275418
  C    3.526001331  -8.667440210  -0.494744304
  H    3.170746943  -9.459310294   0.214008086
  C    3.998610116  -6.393090487   0.313965763
  H    4.612430741  -5.552129451  -0.089169766
  C    4.638390312  -7.767769805   0.050117447
  H    5.489957682  -7.674225672  -0.654867896
  H    5.097105777  -8.175279200   0.974035881
  O    3.933641765  -6.067215214   1.687797953
  O    0.387633731  -5.524620918  -4.100715396
  O    0.417751387  -7.716251141  -2.591576362
  N    3.900141747  -9.332140981  -1.790560888
  C    3.646116169  -8.729655027  -3.087636977
  N    4.088013779  -9.372030745  -4.223851402
  C    4.759487292 -10.567832881  -4.146665525
  C    5.009430268 -11.183405898  -2.861207503
  C    4.577473291 -10.556769383  -1.728393326
  O    3.021160240  -7.674987720  -3.148710321
  N    5.162531213 -11.123752090  -5.317855927
  H    5.654167899 -12.004749679  -5.355729405
  H    4.938087771 -10.678494450  -6.215424716
  H    1.809538153  -4.361307661  -4.343856812
  H    4.747072583 -10.991204618  -0.731202572
  H    5.530933654 -12.140005320  -2.805161966
  H    3.473597757  -6.751256958   2.223610189
  H   -3.430947899  -3.208881043  -4.834808781
  H   -2.614831609  -0.690655147  -0.216463367
  H    1.402878591  -7.959228515  -2.308152346
  H    3.480013649  -8.296166983  -5.646888458
  H   -5.373084366   2.257238447  -4.270657359

