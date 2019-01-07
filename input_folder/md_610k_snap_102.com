%mem=64gb
%nproc=28       
%Chk=snap_102.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_102 

2     1 
  O    3.252532894  -5.048099204  -6.684607597
  C    3.598226819  -3.705242672  -6.352102492
  H    4.632017120  -3.614980288  -6.751415958
  H    3.623746920  -3.593368486  -5.241558200
  C    2.670765171  -2.657226835  -6.974169753
  H    3.052383687  -1.619153531  -6.798792621
  O    2.716061695  -2.786164385  -8.431760686
  C    1.465993687  -3.248266049  -8.929739803
  H    1.367648876  -2.822068808  -9.957711095
  C    1.175374308  -2.768754079  -6.572016316
  H    0.958053692  -3.640516600  -5.911325132
  C    0.408285010  -2.852906401  -7.903871562
  H   -0.020991682  -1.861558517  -8.161734740
  H   -0.468793379  -3.534987628  -7.844356813
  O    0.876879103  -1.511414769  -5.936639894
  N    1.632048121  -4.749399652  -9.073057878
  C    0.713453039  -5.791295430  -8.807409002
  C    1.452540074  -7.010161637  -8.890537340
  N    2.791472371  -6.695292003  -9.175642576
  C    2.900011767  -5.344010102  -9.247852252
  N   -0.612957771  -5.691291518  -8.516585972
  C   -1.233642521  -6.888897218  -8.233425822
  N   -0.579575838  -8.137891878  -8.315658248
  C    0.836724586  -8.286894646  -8.612234691
  N   -2.573740716  -6.859944622  -7.938566216
  H   -3.030864028  -5.942586293  -7.790697669
  H   -3.011255394  -7.630374950  -7.462105544
  O    1.313905715  -9.391653769  -8.579873709
  H    3.813350941  -4.757393671  -9.401905997
  H   -1.064521284  -8.991707750  -8.017052169
  P   -0.106601433  -1.551375235  -4.603309791
  O   -0.423325906   0.069155634  -4.648773189
  C    0.186867218   0.942762912  -3.672212047
  H    0.794674097   1.655700854  -4.279496958
  H    0.845491461   0.391856853  -2.970787445
  C   -0.931439674   1.680490439  -2.937012620
  H   -0.626273163   1.929964519  -1.888625518
  O   -1.105475987   2.971262526  -3.604019999
  C   -2.491029294   3.328962275  -3.673781767
  H   -2.572142448   4.335613346  -3.190830665
  C   -2.334079357   1.016951205  -2.959770626
  H   -2.477502450   0.335570051  -3.836999382
  C   -3.301713925   2.208093661  -3.013472127
  H   -3.613652937   2.482218703  -1.976473395
  H   -4.251738683   1.972679643  -3.525797693
  O   -2.534179802   0.337987463  -1.724005002
  O    0.563853977  -2.160244672  -3.442159295
  O   -1.440590938  -2.047935087  -5.400535559
  N   -2.830257800   3.496567233  -5.121629997
  C   -2.747708429   2.580807542  -6.166943284
  C   -3.253381061   3.238664009  -7.337289069
  N   -3.624438800   4.552390724  -7.014184497
  C   -3.368611449   4.704073903  -5.717215282
  N   -2.321824727   1.258007723  -6.228626851
  C   -2.391358112   0.630679758  -7.480638052
  N   -2.856753906   1.195336678  -8.586691027
  C   -3.306891596   2.532458078  -8.580848870
  H   -2.046320843  -0.427952264  -7.519562782
  N   -3.762561700   3.057172921  -9.731197218
  H   -3.778455616   2.516054099 -10.594639330
  H   -4.099308296   4.020182091  -9.770799730
  H   -3.528631962   5.606663014  -5.130803630
  P   -2.324257526  -1.315532821  -1.786605921
  O   -1.212607850  -1.356501730  -0.603026353
  C   -0.423624194  -2.580180942  -0.531396345
  H   -0.010435091  -2.835020708  -1.534393230
  H    0.407419722  -2.301859025   0.155003827
  C   -1.354819659  -3.632444400   0.060856590
  H   -1.588991285  -3.447868550   1.137018322
  O   -2.615510822  -3.376450976  -0.648299855
  C   -3.201312205  -4.617672019  -1.140678601
  H   -4.233422536  -4.614229901  -0.707986788
  C   -0.999104522  -5.111504591  -0.204745334
  H   -0.178523754  -5.239021903  -0.968648178
  C   -2.316999338  -5.762466384  -0.652644829
  H   -2.139155818  -6.555144901  -1.407019266
  H   -2.777646731  -6.295826723   0.213865906
  O   -0.682330226  -5.740691614   1.053739707
  O   -1.913119662  -1.758283917  -3.162660221
  O   -3.866345458  -1.665144004  -1.381371356
  N   -3.296425969  -4.497227752  -2.621950554
  C   -2.218701577  -4.885873371  -3.474919426
  N   -2.520046378  -4.929396610  -4.858863795
  C   -3.785087846  -4.563310957  -5.435584024
  C   -4.729594627  -3.942270376  -4.498751750
  C   -4.476035760  -3.928914411  -3.167427577
  O   -1.114394551  -5.188126765  -3.062256638
  H   -1.812928068  -5.397730127  -5.462735551
  O   -3.934271381  -4.774033190  -6.622938071
  C   -5.979859775  -3.378170350  -5.077554581
  H   -6.145708752  -2.333492987  -4.778629365
  H   -5.969889563  -3.400302309  -6.181902751
  H   -6.868943590  -3.955434632  -4.770786187
  H   -5.170814532  -3.472378665  -2.443994310
  P    0.915676154  -5.847801318   1.419723493
  O    1.414174057  -6.458181761  -0.032855681
  C    2.838706463  -6.539168853  -0.268452858
  H    3.439071611  -6.202006024   0.599310985
  H    3.031736011  -7.625200006  -0.433346716
  C    3.182336859  -5.734831111  -1.526367147
  H    4.229796535  -5.343651540  -1.474530901
  O    3.210760525  -6.694403461  -2.639822804
  C    2.512954252  -6.195748790  -3.780691890
  H    3.216527385  -6.335066688  -4.641176941
  C    2.196226390  -4.624725097  -1.948904505
  H    1.201183530  -4.718277264  -1.445931107
  C    2.071776808  -4.766605032  -3.473479261
  H    1.032463843  -4.541240950  -3.800721801
  H    2.726725477  -4.024725120  -3.984963368
  O    2.787262770  -3.376800162  -1.590996122
  O    1.313585900  -6.488872195   2.659547842
  O    1.335196357  -4.256102851   1.235839327
  N    1.310826522  -7.086439348  -4.028386992
  C    0.410888302  -6.753144301  -5.120764779
  N   -0.688579383  -7.554256811  -5.358737184
  C   -0.937629523  -8.623650615  -4.532205288
  C   -0.093174609  -8.928278270  -3.417280351
  C    1.010781023  -8.143160803  -3.183447649
  O    0.662138553  -5.755444847  -5.801598935
  N   -2.059952886  -9.363629681  -4.818399016
  H   -2.266299756 -10.221350756  -4.326372986
  H   -2.609998992  -9.161186539  -5.633502501
  H    2.337173882  -5.293480729  -6.359076326
  H    1.698733902  -8.321199276  -2.336859247
  H   -0.311297169  -9.773297143  -2.765828978
  H    2.286995820  -2.643378685  -2.037706744
  H   -2.065569220  -2.665451067  -4.914110567
  H   -4.064426043  -1.777301849  -0.398951112
  H    1.946294969  -3.844692321   1.898825253
  H    3.529016040  -7.388455714  -9.273732936
  H   -1.838464476   0.792896385  -5.421100255
