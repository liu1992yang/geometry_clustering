%mem=64gb
%nproc=28       
%Chk=snap_86.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_86 

2     1 
  O    2.824771770  -4.775751758  -7.467687976
  C    2.520618303  -3.556958914  -8.156103155
  H    3.090545139  -3.674264118  -9.108207366
  H    2.929761251  -2.707631743  -7.576241364
  C    1.036658694  -3.345380358  -8.459212333
  H    0.886788489  -2.333943960  -8.917916152
  O    0.659123938  -4.281090227  -9.527740939
  C   -0.552831544  -4.951434712  -9.215275668
  H   -1.130169954  -5.037068340 -10.165982923
  C    0.009038059  -3.559472302  -7.317820280
  H    0.405515778  -4.180794073  -6.478359243
  C   -1.190697882  -4.211203609  -8.035888613
  H   -1.907752935  -3.435799652  -8.384114988
  H   -1.801745688  -4.857216800  -7.368901779
  O   -0.310577999  -2.234930330  -6.867034343
  N   -0.187231648  -6.350339957  -8.762623720
  C   -1.113571626  -7.403031802  -8.583661639
  C   -0.521744762  -8.309991547  -7.655671679
  N    0.752880046  -7.816710868  -7.320436468
  C    0.933289129  -6.641049283  -7.961692581
  N   -2.327612109  -7.575014126  -9.186812212
  C   -3.014368805  -8.688964122  -8.774384947
  N   -2.527871956  -9.601483543  -7.802921816
  C   -1.246345302  -9.438599257  -7.141250143
  N   -4.252366772  -8.913496817  -9.326801637
  H   -4.577006023  -8.302661708 -10.074333473
  H   -4.759769320  -9.775337269  -9.191868585
  O   -0.948734340 -10.224544571  -6.267349669
  H    1.862391260  -5.946919552  -7.909104147
  H   -3.114709661 -10.380501453  -7.473481155
  P   -1.017535400  -2.126930910  -5.363686114
  O   -1.858445437  -0.730974715  -5.679218278
  C   -1.217063007   0.509856202  -5.359020014
  H   -1.567598915   1.185261554  -6.168988364
  H   -0.111982126   0.449051091  -5.387978542
  C   -1.730473180   0.980230536  -3.985881126
  H   -1.041818488   0.709306568  -3.146936464
  O   -1.681098373   2.444660182  -3.986926761
  C   -2.972216061   3.015078744  -3.743261958
  H   -2.836860294   3.690364156  -2.858039224
  C   -3.202343051   0.572983840  -3.720755423
  H   -3.615353626  -0.097439637  -4.517542900
  C   -3.985738529   1.887831095  -3.570800700
  H   -4.451783332   1.911559472  -2.552943912
  H   -4.836858262   1.924404507  -4.276396180
  O   -3.350247682  -0.050877350  -2.436238718
  O   -0.000517269  -2.099141987  -4.293422117
  O   -2.130677929  -3.294176221  -5.576389534
  N   -3.277794323   3.878754776  -4.935798935
  C   -2.374645138   4.368425967  -5.869650514
  C   -3.109768217   5.228097915  -6.751576555
  N   -4.452528426   5.266673121  -6.351275279
  C   -4.547754024   4.479068517  -5.280283193
  N   -1.005137480   4.159958259  -6.030041143
  C   -0.395868122   4.850740389  -7.093402718
  N   -1.033186073   5.649970658  -7.933595786
  C   -2.419281487   5.896909252  -7.813092976
  H    0.698116660   4.701739517  -7.234153941
  N   -2.991297590   6.737055761  -8.689098474
  H   -2.448978779   7.193608441  -9.424364891
  H   -3.988728154   6.950789704  -8.641283088
  H   -5.443993776   4.293976162  -4.693592937
  P   -2.722559727  -1.571911263  -2.243317845
  O   -1.451713836  -1.072769107  -1.350041405
  C   -0.337220926  -2.004186253  -1.260653034
  H   -0.022764793  -2.329669671  -2.283705694
  H    0.465392714  -1.378634978  -0.815733174
  C   -0.823205308  -3.118499725  -0.339988230
  H   -0.999232056  -2.760139819   0.703248978
  O   -2.160718689  -3.415677709  -0.882309988
  C   -2.363628123  -4.858609974  -1.023628973
  H   -3.368749495  -5.017006985  -0.559014716
  C   -0.067883473  -4.463078558  -0.340969116
  H    0.580226506  -4.595604575  -1.256095361
  C   -1.201906620  -5.507154060  -0.276773318
  H   -0.890286426  -6.494470017  -0.688419336
  H   -1.484067056  -5.736144169   0.775040007
  O    0.771698119  -4.432914943   0.821072460
  O   -2.444846542  -2.234227844  -3.554893825
  O   -3.977483199  -2.169192882  -1.400796699
  N   -2.443563562  -5.195507094  -2.472086988
  C   -1.298297183  -5.628612096  -3.216891698
  N   -1.530978143  -6.122905221  -4.514999937
  C   -2.807039078  -6.151551697  -5.166000434
  C   -3.899617918  -5.544743855  -4.382960783
  C   -3.704114532  -5.104957343  -3.114858313
  O   -0.171459560  -5.614390576  -2.751645677
  H   -0.688499524  -6.478328628  -5.009779782
  O   -2.852239720  -6.642905633  -6.274430891
  C   -5.217381434  -5.445482327  -5.067731712
  H   -5.780811883  -6.390313314  -4.982954021
  H   -5.854550543  -4.647710717  -4.662519425
  H   -5.099076617  -5.246228337  -6.147445031
  H   -4.512164126  -4.644476107  -2.525688039
  P    1.679223410  -5.805112647   1.080220350
  O    2.856922071  -5.719265362  -0.045673180
  C    4.026953617  -4.886671571   0.173162468
  H    3.956189323  -4.273281694   1.095503434
  H    4.861364391  -5.615724790   0.282070097
  C    4.223686804  -4.015719205  -1.071655764
  H    5.167699623  -3.423096448  -0.965379082
  O    4.491906964  -4.855288131  -2.230376460
  C    3.436057798  -4.771945546  -3.197566177
  H    3.947365066  -4.533471868  -4.161460332
  C    3.009163115  -3.122014217  -1.433125723
  H    2.287345481  -3.027773647  -0.589353809
  C    2.420756508  -3.736769367  -2.708276539
  H    1.426076986  -4.214857476  -2.512738379
  H    2.175877507  -2.973699338  -3.480112810
  O    3.402025192  -1.770255584  -1.613278775
  O    0.929371133  -7.059856854   1.057788828
  O    2.450249390  -5.301088295   2.426379677
  N    2.825312838  -6.144171973  -3.341681904
  C    1.912183052  -6.380629777  -4.439193511
  N    1.365251781  -7.636399135  -4.601158190
  C    1.548994695  -8.604490093  -3.635401633
  C    2.388729799  -8.352656664  -2.498332577
  C    3.017663432  -7.139144898  -2.385863449
  O    1.658319348  -5.459077686  -5.220934245
  N    0.905375372  -9.787248843  -3.846454118
  H    0.991144611 -10.563281798  -3.206525570
  H    0.311328647  -9.927704013  -4.667848428
  H    2.569308274  -4.728905454  -6.493518363
  H    3.709783973  -6.905052885  -1.555774037
  H    2.519932382  -9.110203196  -1.725122770
  H    4.030507310  -1.658382715  -2.360232759
  H   -2.740940115  -3.472161729  -4.767610651
  H   -3.865994158  -2.255099885  -0.403286220
  H    2.100424205  -5.584729913   3.316827039
  H    1.308950338  -8.178650087  -6.532213645
  H   -0.505370052   3.561293470  -5.352677040
