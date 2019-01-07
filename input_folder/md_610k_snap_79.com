%mem=64gb
%nproc=28       
%Chk=snap_79.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_79 

2     1 
  O    3.282128413  -5.542278619  -7.041558571
  C    3.273839344  -4.210905341  -7.578123368
  H    3.888957553  -4.324603727  -8.502369616
  H    3.805369304  -3.549212225  -6.866842872
  C    1.893262799  -3.649790099  -7.925844075
  H    1.972133646  -2.562233746  -8.190624046
  O    1.456794307  -4.270289481  -9.176604498
  C    0.193845367  -4.902778955  -9.033538931
  H   -0.387040816  -4.651548342  -9.957584926
  C    0.758599402  -3.863053570  -6.894501590
  H    1.064422938  -4.509807543  -6.030315992
  C   -0.405621903  -4.468718102  -7.696601056
  H   -1.223610307  -3.730397267  -7.851240412
  H   -0.893117877  -5.299947571  -7.138393335
  O    0.488363719  -2.535984255  -6.401049466
  N    0.438009381  -6.392859437  -9.058075289
  C   -0.407856226  -7.354783040  -9.652266236
  C    0.150504574  -8.629128867  -9.353712066
  N    1.314648978  -8.417932981  -8.586447619
  C    1.475389811  -7.081239436  -8.415734566
  N   -1.536394785  -7.128491390 -10.389241889
  C   -2.148749518  -8.266958057 -10.858985350
  N   -1.670403849  -9.570995150 -10.599574830
  C   -0.471370786  -9.848672037  -9.811065254
  N   -3.285822882  -8.107154297 -11.607933089
  H   -3.618993156  -7.168628431 -11.817086220
  H   -3.772654933  -8.878313785 -12.039997349
  O   -0.161487904 -10.999777528  -9.646119529
  H    2.324146328  -6.534969117  -7.853260007
  H   -2.169840544 -10.397391028 -10.960936882
  P   -0.571864116  -2.362897120  -5.131242960
  O   -1.501349129  -1.150807084  -5.760569230
  C   -1.090155153   0.191882042  -5.459598978
  H   -1.657408217   0.783330853  -6.210895259
  H   -0.005240384   0.340936909  -5.633982771
  C   -1.481562463   0.536829189  -4.016752825
  H   -0.931684807  -0.103987862  -3.275564189
  O   -0.951098147   1.879699142  -3.747132360
  C   -1.848417598   2.609971342  -2.888962011
  H   -1.261636378   2.851198063  -1.964745064
  C   -3.006961602   0.640546420  -3.748638673
  H   -3.582118053   0.886190493  -4.677639060
  C   -3.099174134   1.762359448  -2.686675163
  H   -3.116060181   1.313580826  -1.664059693
  H   -4.055132845   2.309901070  -2.763857254
  O   -3.688935140  -0.507797251  -3.283528409
  O    0.186475831  -2.020758148  -3.899378693
  O   -1.400708761  -3.720758830  -5.419004802
  N   -2.125799531   3.896501093  -3.609751233
  C   -1.375031252   4.457897740  -4.635256502
  C   -1.953172898   5.738064963  -4.921864085
  N   -3.046227770   5.956623631  -4.071997444
  C   -3.143062780   4.876796661  -3.297611010
  N   -0.260020810   3.991894489  -5.330738726
  C    0.266540506   4.857680848  -6.306886087
  N   -0.233463344   6.045138008  -6.605983449
  C   -1.363871930   6.562404336  -5.934164378
  H    1.163765150   4.507266262  -6.864108901
  N   -1.804959652   7.780416212  -6.281660088
  H   -1.337482779   8.331634629  -7.003588634
  H   -2.618390190   8.196758744  -5.825117167
  H   -3.871288331   4.713092142  -2.507136876
  P   -2.978930155  -1.839155813  -2.581402067
  O   -2.148121105  -0.959715425  -1.479033954
  C   -0.973574511  -1.624452438  -0.930752346
  H   -0.277759136  -1.905280601  -1.753555760
  H   -0.499798974  -0.831007135  -0.313668932
  C   -1.478146565  -2.796237330  -0.098142692
  H   -1.825117176  -2.484302970   0.915411974
  O   -2.688112629  -3.250525108  -0.799746053
  C   -2.808969492  -4.715361120  -0.701727776
  H   -3.786518672  -4.843436207  -0.169763963
  C   -0.572994276  -4.049772328  -0.041718472
  H    0.061807765  -4.154169324  -0.964103375
  C   -1.591502855  -5.199136476   0.080826942
  H   -1.173863926  -6.164008422  -0.268373529
  H   -1.828437282  -5.370434692   1.156595406
  O    0.198377509  -4.095422738   1.150415900
  O   -2.185662851  -2.667557344  -3.534430650
  O   -4.412098011  -2.450787399  -2.105208473
  N   -2.933855371  -5.257533274  -2.077905762
  C   -1.826920886  -5.788147720  -2.808198321
  N   -2.138819888  -6.383724373  -4.045101367
  C   -3.447310389  -6.426430005  -4.662305597
  C   -4.485698614  -5.717226218  -3.897447673
  C   -4.219612850  -5.182790212  -2.683375826
  O   -0.673163910  -5.751766751  -2.404665144
  H   -1.363075025  -6.875558238  -4.537347027
  O   -3.527076071  -7.003024558  -5.719802981
  C   -5.836063572  -5.646537803  -4.520036073
  H   -6.479816377  -6.471199197  -4.169549155
  H   -6.355699568  -4.702778956  -4.302618217
  H   -5.786194434  -5.737991534  -5.619209981
  H   -4.988493526  -4.649770653  -2.099741069
  P    1.591225463  -3.188314703   1.265682761
  O    2.797038886  -4.220716872   0.838209944
  C    2.606698754  -5.267786470  -0.122776536
  H    3.281493929  -6.070353925   0.266412055
  H    1.578496800  -5.665360984  -0.144026533
  C    3.132061635  -4.759335321  -1.470843505
  H    4.029941634  -4.104589468  -1.319124174
  O    3.642293476  -5.933655502  -2.179909682
  C    3.147231577  -5.992803005  -3.521790763
  H    4.035462322  -6.249396237  -4.149798331
  C    2.115296699  -4.093427218  -2.424529771
  H    1.040349035  -4.270811206  -2.149822641
  C    2.472055733  -4.649673384  -3.809598740
  H    1.581437276  -4.725231089  -4.469721341
  H    3.179254331  -3.956739335  -4.314455451
  O    2.391168229  -2.685222151  -2.371253747
  O    1.772070717  -2.540258453   2.548345731
  O    1.379146548  -2.241835861  -0.055047662
  N    2.174816522  -7.152188428  -3.582158936
  C    1.245321787  -7.300917866  -4.692647385
  N    0.331006659  -8.323381403  -4.677636491
  C    0.295775011  -9.204283773  -3.622091564
  C    1.226523049  -9.095695747  -2.530523398
  C    2.133429599  -8.068448208  -2.533531386
  O    1.309247995  -6.489820123  -5.622163175
  N   -0.682735428 -10.156009993  -3.661421450
  H   -0.739528596 -10.888945956  -2.970134744
  H   -1.302610360 -10.225038601  -4.462353765
  H    2.695504582  -5.628827355  -6.229301643
  H    2.870280708  -7.930503379  -1.721378249
  H    1.211751295  -9.818764532  -1.715443340
  H    1.728001603  -2.193852985  -2.968711592
  H   -1.929725031  -4.143465130  -4.673841269
  H   -5.089420187  -1.787551882  -1.764475182
  H    2.087479239  -2.226683449  -0.812943296
  H    1.895767423  -9.165372608  -8.223516665
  H    0.143392422   3.076100132  -5.080886168

