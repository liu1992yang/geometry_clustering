%mem=64gb
%nproc=28       
%Chk=snap_37.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_37 

2     1 
  O    2.139161544  -4.425372877  -7.524031799
  C    2.376102778  -3.226371348  -8.284824461
  H    3.337179544  -3.450688443  -8.801415116
  H    2.536874988  -2.389116982  -7.572563644
  C    1.276982729  -2.904409193  -9.297744636
  H    1.420198352  -1.872015222  -9.711131730
  O    1.464848238  -3.766459507 -10.469249220
  C    0.274677972  -4.468771035 -10.791184099
  H    0.201626544  -4.474054289 -11.908298606
  C   -0.183112969  -3.137975223  -8.826638137
  H   -0.254797102  -3.775878757  -7.903265208
  C   -0.874866846  -3.818995436 -10.015373852
  H   -1.408601842  -3.062640570 -10.635502448
  H   -1.657421627  -4.532422253  -9.696206603
  O   -0.856491827  -1.877659234  -8.622502595
  N    0.484211431  -5.899855536 -10.345927337
  C   -0.001665494  -7.050083680 -11.005681380
  C    0.424720485  -8.170983712 -10.240175473
  N    1.152179697  -7.683669205  -9.134712836
  C    1.181019201  -6.329661110  -9.207614402
  N   -0.741072123  -7.101350645 -12.153827574
  C   -1.056693688  -8.374075820 -12.573832385
  N   -0.661450358  -9.540290507 -11.882356934
  C    0.117501511  -9.522818181 -10.643967957
  N   -1.812769644  -8.491559878 -13.711218618
  H   -2.071757206  -7.654536468 -14.229931899
  H   -2.041536501  -9.381150339 -14.130969081
  O    0.392269533 -10.578031457 -10.138224916
  H    1.690450841  -5.590735541  -8.462618111
  H   -0.914411215 -10.475051537 -12.237309019
  P   -0.506172844  -1.171938061  -7.172040396
  O   -1.924924978  -0.377723798  -6.841439196
  C   -1.805740086   1.047492419  -6.660960691
  H   -2.844698124   1.403204480  -6.825501607
  H   -1.140336925   1.513530926  -7.415790540
  C   -1.319192833   1.309479809  -5.226190518
  H   -0.218578846   1.132013610  -5.085444798
  O   -1.454678778   2.736847527  -4.954922526
  C   -2.404067199   2.982093965  -3.907115111
  H   -1.842643387   3.554121619  -3.121522242
  C   -2.167364619   0.560489518  -4.168906225
  H   -2.802864788  -0.242278798  -4.612879022
  C   -2.991585741   1.648016847  -3.452851940
  H   -2.907723370   1.519355276  -2.347956840
  H   -4.069661840   1.528796653  -3.673956359
  O   -1.188624001   0.039874256  -3.257228492
  O    0.679024659  -0.303802520  -7.146124221
  O   -0.538377191  -2.454551461  -6.204383974
  N   -3.416859679   3.910235921  -4.516372473
  C   -3.176256842   4.776740493  -5.577654803
  C   -4.335834770   5.608401041  -5.709166891
  N   -5.277298340   5.251159420  -4.733684066
  C   -4.732638544   4.258338931  -4.029737491
  N   -2.078633494   4.915536406  -6.426019032
  C   -2.172114295   5.929758565  -7.394916992
  N   -3.217233992   6.727903561  -7.547715662
  C   -4.353049150   6.628282728  -6.714874123
  H   -1.302396872   6.059827110  -8.078360436
  N   -5.361484572   7.491949279  -6.912699305
  H   -5.318935690   8.206204920  -7.641462997
  H   -6.199240867   7.465803927  -6.330448170
  H   -5.182471419   3.748600267  -3.181801638
  P   -1.430016663  -1.471971521  -2.632871000
  O   -0.459336606  -1.267109752  -1.351567161
  C    0.071051975  -2.479208309  -0.748566196
  H    0.721510817  -3.017901421  -1.477356861
  H    0.710506289  -2.075418590   0.067325319
  C   -1.114014006  -3.292558797  -0.230370294
  H   -1.653249175  -2.793084957   0.610474992
  O   -2.080868783  -3.287237373  -1.340866454
  C   -2.390615702  -4.664259020  -1.771881092
  H   -3.504755248  -4.649267202  -1.847015234
  C   -0.811417010  -4.773264300   0.094809578
  H    0.252991526  -5.060822603  -0.145238610
  C   -1.828932870  -5.589547009  -0.707408964
  H   -1.353111467  -6.516226429  -1.137589383
  H   -2.616035861  -5.996712427  -0.036195219
  O   -1.139879695  -4.909484080   1.493978174
  O   -1.148118971  -2.492509090  -3.686561744
  O   -2.998721322  -1.282342190  -2.202231331
  N   -1.801472274  -4.895402390  -3.114944458
  C   -0.378697031  -4.983063896  -3.286038144
  N    0.064047618  -5.249367652  -4.612010503
  C   -0.773291617  -5.250101229  -5.756116403
  C   -2.188056989  -4.968089751  -5.512053291
  C   -2.642452876  -4.776763708  -4.245559106
  O    0.427827971  -4.875098159  -2.384514822
  H    1.077579697  -5.396014186  -4.730664482
  O   -0.225849561  -5.459149637  -6.840396276
  C   -3.057777265  -4.833398615  -6.713325960
  H   -4.124692800  -4.969021359  -6.489836284
  H   -2.939431465  -3.834868823  -7.167686121
  H   -2.793877508  -5.578085888  -7.481625662
  H   -3.694117767  -4.537935752  -4.041435807
  P   -0.098498730  -5.703670492   2.484296886
  O    1.210006632  -5.756296369   1.506172040
  C    2.537940543  -5.850166588   2.092195510
  H    2.694486784  -4.983637866   2.766076289
  H    2.626305871  -6.810249015   2.642246526
  C    3.506732408  -5.828501351   0.899796946
  H    4.462136659  -5.332755507   1.187148109
  O    3.907184358  -7.194475440   0.611030220
  C    3.260208412  -7.662231596  -0.598138697
  H    3.970285170  -8.412424195  -1.016681660
  C    2.913444732  -5.261205691  -0.425330020
  H    1.849027942  -4.920040700  -0.322247113
  C    3.047174896  -6.408026868  -1.442617883
  H    2.144856381  -6.457642363  -2.103119619
  H    3.895243714  -6.249516489  -2.131736450
  O    3.573042051  -4.071144618  -0.815789780
  O   -0.592346028  -6.922657644   3.127784165
  O    0.325650186  -4.501687831   3.518094442
  N    2.012508614  -8.362132031  -0.139588726
  C    0.658458932  -8.008710343  -0.585213717
  N   -0.404321305  -8.292365498   0.257346776
  C   -0.226783239  -9.035418979   1.385425418
  C    1.078835177  -9.544810614   1.744785244
  C    2.158196375  -9.157553445   1.006814137
  O    0.512172738  -7.457361496  -1.665904557
  N   -1.344789620  -9.336606162   2.136023703
  H   -1.218242909  -9.547525910   3.119879465
  H   -2.194106763  -8.812035974   1.945888822
  H    1.284178216  -4.361459824  -7.002053294
  H    3.185459850  -9.443160269   1.283514695
  H    1.186425407 -10.215430006   2.594884725
  H    4.516928025  -4.218895332  -1.041044152
  H   -0.784134084  -2.340545957  -5.163234749
  H   -3.371804917  -1.897917703  -1.493569335
  H   -0.036280647  -4.528964623   4.444630769
  H    1.556440483  -8.277417609  -8.417525384
  H   -1.255177198   4.302618661  -6.295514099
