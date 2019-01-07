%mem=64gb
%nproc=28       
%Chk=snap_193.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_193 

2     1 
  O    1.924306274  -2.864894182  -4.363965015
  C    2.544433259  -2.774779909  -5.642273658
  H    3.238299866  -3.640453410  -5.654973105
  H    3.140258082  -1.843502037  -5.700160481
  C    1.532183317  -2.859960625  -6.796039224
  H    1.414367008  -1.885472931  -7.327304199
  O    2.110039443  -3.737050470  -7.812985643
  C    1.231441640  -4.805954605  -8.144640374
  H    1.222391042  -4.854264144  -9.265421658
  C    0.152910524  -3.455977335  -6.432670426
  H    0.106966264  -3.840803481  -5.381057190
  C   -0.109337078  -4.570038061  -7.455687304
  H   -0.887726912  -4.238706591  -8.181508596
  H   -0.542249587  -5.471625874  -6.974335228
  O   -0.807696499  -2.404765270  -6.709543298
  N    1.896969913  -6.068872685  -7.656181712
  C    2.545153700  -7.003528642  -8.487904704
  C    3.131512562  -7.984323355  -7.632569783
  N    2.837864408  -7.619917489  -6.304597688
  C    2.105077756  -6.480278505  -6.324828675
  N    2.584681663  -7.004185477  -9.854285118
  C    3.267789292  -8.056837756 -10.411060283
  N    3.886184144  -9.068732088  -9.638183386
  C    3.859207737  -9.100293327  -8.180485183
  N    3.336435619  -8.120050117 -11.777044987
  H    2.897773845  -7.389931551 -12.335785540
  H    3.827826619  -8.848655253 -12.273355441
  O    4.427691723 -10.028636474  -7.657262153
  H    1.665453867  -5.946658568  -5.432494577
  H    4.392239081  -9.843172396 -10.091918918
  P   -1.663424795  -1.901284015  -5.399610077
  O   -2.629909018  -0.743934876  -6.050893639
  C   -1.961890596   0.451626144  -6.515658719
  H   -2.784841863   0.985934808  -7.037855261
  H   -1.186110412   0.200927248  -7.266764058
  C   -1.406746976   1.274779917  -5.343051468
  H   -0.335075357   1.050279232  -5.107409805
  O   -1.379647474   2.664556879  -5.773683623
  C   -2.265159169   3.468482750  -4.992878035
  H   -1.698715611   4.417880980  -4.798233463
  C   -2.288440198   1.215578283  -4.067819323
  H   -3.196754163   0.582067645  -4.210608460
  C   -2.644055054   2.678029180  -3.747151681
  H   -2.022305812   3.004869411  -2.871777380
  H   -3.678347522   2.805272872  -3.380373222
  O   -1.458353762   0.753949916  -2.994504634
  O   -0.660625945  -1.434075668  -4.357953314
  O   -2.588897449  -3.190221337  -5.170993539
  N   -3.394239669   3.823699937  -5.917788243
  C   -4.731426617   3.444049445  -5.983353110
  C   -5.284321951   4.082980064  -7.145684405
  N   -4.283871116   4.808526280  -7.809542332
  C   -3.174860171   4.645053947  -7.094366565
  N   -5.536425728   2.648153266  -5.175831841
  C   -6.897091501   2.572734830  -5.515344761
  N   -7.450640544   3.156054766  -6.568421745
  C   -6.676526662   3.943848389  -7.443535517
  H   -7.541441085   1.966764000  -4.837766085
  N   -7.288191404   4.515657669  -8.495204522
  H   -8.287590795   4.399112337  -8.657498334
  H   -6.767538062   5.098997982  -9.149527939
  H   -2.189606160   5.052018738  -7.316798275
  P   -1.765501765  -0.794554497  -2.485218067
  O   -0.341645514  -1.293044764  -1.887779598
  C   -0.044960765  -2.702729972  -1.993700823
  H    0.246287240  -2.938808206  -3.042140208
  H    0.863791852  -2.779018993  -1.355097537
  C   -1.193363176  -3.556854278  -1.450977914
  H   -0.893343886  -4.083338986  -0.512229939
  O   -2.250101366  -2.615819423  -1.091620278
  C   -3.537533394  -3.351696623  -1.061024531
  H   -3.705270187  -3.582936054   0.021320821
  C   -1.889378669  -4.544083955  -2.427139650
  H   -1.830168807  -4.263663319  -3.505460701
  C   -3.338949851  -4.601911271  -1.919834660
  H   -4.070027804  -4.699929249  -2.756069838
  H   -3.490991022  -5.535187715  -1.328170361
  O   -1.364745483  -5.878940195  -2.228280380
  O   -2.754242503  -1.483757480  -3.423150787
  O   -2.497257928  -0.307588938  -1.085954546
  N   -4.564115203  -2.363139060  -1.486435480
  C   -4.858203583  -2.162057852  -2.873305071
  N   -5.335251431  -0.874270667  -3.257049127
  C   -5.435350470   0.228591638  -2.367650947
  C   -5.315084533  -0.098488774  -0.947186711
  C   -4.831968436  -1.308978551  -0.564500112
  O   -4.853770519  -3.069211140  -3.693787547
  H   -5.450207150  -0.724472659  -4.268562808
  O   -5.603600793   1.341212483  -2.864256125
  C   -5.677768303   0.964477326   0.029265853
  H   -4.837690173   1.212837988   0.697104538
  H   -5.964692185   1.904546606  -0.472852310
  H   -6.531838425   0.669435965   0.659337922
  H   -4.641151353  -1.546820183   0.491341006
  P    0.100526997  -6.154964941  -2.909939861
  O    0.998564563  -5.273084511  -1.835066418
  C    2.405082159  -5.556858499  -1.701250386
  H    2.854662694  -5.967716744  -2.628657663
  H    2.834693458  -4.544643917  -1.523003650
  C    2.589798071  -6.457268148  -0.476104138
  H    1.833947557  -6.200319438   0.307326032
  O    2.256072518  -7.844609834  -0.795887783
  C    3.361778149  -8.731382547  -0.558982006
  H    2.988934502  -9.422876087   0.239871373
  C    4.041521446  -6.455212240   0.073159151
  H    4.694153671  -5.675605783  -0.387679063
  C    4.570451199  -7.881626487  -0.156264557
  H    5.363606122  -7.883571556  -0.931857656
  H    5.077571845  -8.269227202   0.750711664
  O    4.057258086  -6.072676726   1.434518380
  O    0.288361605  -5.700219295  -4.312704891
  O    0.320442736  -7.722634971  -2.624101454
  N    3.597727921  -9.530648842  -1.805549651
  C    3.489632515  -8.964667085  -3.139360609
  N    3.867274914  -9.715046170  -4.229114845
  C    4.335477800 -10.995248376  -4.071401481
  C    4.389029605 -11.598946231  -2.755446878
  C    4.030599621 -10.857398292  -1.668316405
  O    3.037857489  -7.829130946  -3.279717803
  N    4.739393013 -11.653633365  -5.188617033
  H    5.092128053 -12.598281414  -5.157978010
  H    4.709062901 -11.193015117  -6.102950680
  H    1.278033473  -2.113730997  -4.203720285
  H    4.074139275 -11.275845531  -0.651200732
  H    4.711698064 -12.634469180  -2.640270876
  H    3.587383778  -6.708583075   2.018505621
  H   -3.473759895  -3.050569774  -4.632107725
  H   -1.970608409  -0.429526419  -0.248312395
  H    1.283878056  -7.990583430  -2.324486717
  H    3.139223616  -8.149534238  -5.464484414
  H   -5.178770514   2.167059795  -4.308081318
