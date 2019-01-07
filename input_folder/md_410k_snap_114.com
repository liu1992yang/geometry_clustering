%mem=64gb
%nproc=28       
%Chk=snap_114.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_114 

2     1 
  O    2.989121434  -2.855279882  -6.540959425
  C    2.616252825  -3.870751433  -7.483802135
  H    2.592091445  -4.841405730  -6.940910983
  H    3.446783150  -3.883569309  -8.219976083
  C    1.275154685  -3.521322395  -8.129792175
  H    1.309177888  -2.534227898  -8.650704984
  O    0.998406144  -4.500155257  -9.182662338
  C   -0.343362225  -4.975839630  -9.089716059
  H   -0.753946464  -4.998489283 -10.128099837
  C    0.069646237  -3.601658674  -7.154122909
  H    0.267442195  -4.278147463  -6.288385418
  C   -1.070476342  -4.124030583  -8.045256889
  H   -1.626023085  -3.284525570  -8.513785868
  H   -1.844927299  -4.680622872  -7.468498336
  O   -0.133873508  -2.253335543  -6.712055597
  N   -0.243654212  -6.405773434  -8.605976767
  C   -1.254966163  -7.384807283  -8.704454312
  C   -0.886370206  -8.443386203  -7.820686472
  N    0.316091849  -8.086266928  -7.192625425
  C    0.688842473  -6.868674585  -7.661710661
  N   -2.364688389  -7.364471201  -9.502535339
  C   -3.184972997  -8.456341045  -9.364188632
  N   -2.910543365  -9.536837227  -8.493625824
  C   -1.740238249  -9.589293368  -7.624805407
  N   -4.350460405  -8.464928642 -10.092507010
  H   -4.518833410  -7.716159683 -10.762251279
  H   -4.939680069  -9.280923734 -10.167089993
  O   -1.622959342 -10.533119949  -6.883493626
  H    1.592785266  -6.302350356  -7.372177373
  H   -3.571958250 -10.321544322  -8.406977407
  P   -0.912401490  -2.069424800  -5.249055565
  O   -1.769514615  -0.715155016  -5.653079304
  C   -1.215785195   0.549422952  -5.257509876
  H   -1.545107121   1.225048677  -6.076492380
  H   -0.109234559   0.544360783  -5.214187285
  C   -1.845228630   0.955644853  -3.912827484
  H   -1.207668081   0.693689089  -3.032048541
  O   -1.863690617   2.416911659  -3.860823629
  C   -3.200702044   2.928173874  -3.854568128
  H   -3.227417713   3.666863588  -3.010793630
  C   -3.314744297   0.479680307  -3.778400668
  H   -3.618503181  -0.227171081  -4.592520017
  C   -4.171361305   1.757571769  -3.744058084
  H   -4.731861780   1.784168122  -2.773969571
  H   -4.955108218   1.730909132  -4.522630269
  O   -3.557826858  -0.124919961  -2.499083128
  O    0.101217688  -1.944001533  -4.179227890
  O   -1.962926581  -3.290435725  -5.439641742
  N   -3.345623066   3.692689478  -5.142556467
  C   -2.310611901   4.299538034  -5.845539963
  C   -2.909020266   5.037343101  -6.917844184
  N   -4.301004235   4.881673213  -6.869028013
  C   -4.555556211   4.096334378  -5.822067783
  N   -0.929406553   4.280666164  -5.656371685
  C   -0.171462028   5.047367879  -6.560071663
  N   -0.680344438   5.747625005  -7.560897450
  C   -2.071890219   5.795492211  -7.798709806
  H    0.932440556   5.056049017  -6.414958695
  N   -2.517542707   6.544413242  -8.818771899
  H   -1.876726221   7.074106232  -9.411668193
  H   -3.515873267   6.618729781  -9.021494307
  H   -5.538733950   3.784632957  -5.478912930
  P   -2.846871099  -1.589380461  -2.198497799
  O   -1.715719892  -1.003340931  -1.183675718
  C   -0.574903848  -1.876962466  -0.948884595
  H   -0.116459346  -2.186973978  -1.918324898
  H    0.140850212  -1.219221167  -0.411130827
  C   -1.101887530  -3.018022553  -0.088768288
  H   -1.435760384  -2.681685157   0.922569341
  O   -2.330511219  -3.421647661  -0.793545774
  C   -2.397713025  -4.879778753  -0.940271188
  H   -3.420044215  -5.122965370  -0.554039471
  C   -0.238253117  -4.288581779   0.005094397
  H    0.530277972  -4.334779923  -0.820719536
  C   -1.257018292  -5.440785743  -0.099606843
  H   -0.790908462  -6.365532268  -0.510417064
  H   -1.610332933  -5.742683674   0.912393096
  O    0.387531557  -4.233813553   1.289098888
  O   -2.361954740  -2.225460295  -3.465172567
  O   -4.146675190  -2.270457248  -1.506214235
  N   -2.344198437  -5.205082309  -2.393176732
  C   -1.125763750  -5.567369701  -3.061212563
  N   -1.251450149  -6.023925924  -4.391296218
  C   -2.476253037  -6.078421334  -5.131968709
  C   -3.660360324  -5.601021534  -4.396860904
  C   -3.567979416  -5.187455882  -3.108181530
  O   -0.036602348  -5.514011096  -2.520296390
  H   -0.372271322  -6.333424464  -4.862427800
  O   -2.419433746  -6.483281486  -6.276431699
  C   -4.950076444  -5.600802235  -5.140499122
  H   -4.816707668  -5.923132954  -6.188536601
  H   -5.678972651  -6.294188938  -4.690877931
  H   -5.414654238  -4.604301438  -5.173242196
  H   -4.442967202  -4.804141745  -2.558834097
  P    1.695383422  -5.254866743   1.455991597
  O    2.826612477  -4.398242127   0.613328571
  C    3.496747306  -5.114557162  -0.450893731
  H    4.490187051  -5.410862610  -0.048253261
  H    2.944010663  -6.023920277  -0.767789501
  C    3.660185496  -4.100846384  -1.593141888
  H    4.275373684  -3.219888308  -1.296729497
  O    4.457991653  -4.809284119  -2.601381998
  C    3.720343709  -4.988053789  -3.811148984
  H    4.466475433  -4.872317428  -4.632516300
  C    2.354503068  -3.671782512  -2.299078188
  H    1.454725396  -4.205474396  -1.888942143
  C    2.575004822  -3.984416765  -3.782724083
  H    1.642323043  -4.337342033  -4.266527781
  H    2.863700749  -3.070823903  -4.352027024
  O    2.157151974  -2.279216581  -2.055800886
  O    1.491484740  -6.642020226   1.035786347
  O    2.078594718  -4.920041134   2.991177801
  N    3.234492023  -6.434160509  -3.817139436
  C    2.133372425  -6.852719598  -4.646533302
  N    1.710309716  -8.168041863  -4.624889307
  C    2.319194092  -9.071288902  -3.784617072
  C    3.433524292  -8.675596268  -2.964159404
  C    3.863841038  -7.376065252  -3.004317123
  O    1.586904509  -6.034130344  -5.399660822
  N    1.831256052 -10.345918143  -3.799527485
  H    2.209350360 -11.067914305  -3.202341077
  H    1.022620182 -10.591004526  -4.358128626
  H    2.194137529  -2.366188687  -6.225947776
  H    4.724316168  -7.018947350  -2.408797244
  H    3.927946821  -9.402533525  -2.318676324
  H    1.666275599  -1.870206877  -2.822650549
  H   -2.575972115  -3.477854661  -4.626326039
  H   -4.112740820  -2.428856852  -0.509022987
  H    2.311520773  -3.968930600   3.217446104
  H    0.758854685  -8.589053710  -6.405436112
  H   -0.534271963   3.738410593  -4.872041105

