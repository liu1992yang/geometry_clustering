%mem=64gb
%nproc=28       
%Chk=snap_135.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_135 

2     1 
  O    1.598920704  -2.221253895  -6.169789198
  C    1.668754075  -3.259979731  -7.139868592
  H    1.788925940  -4.244143091  -6.641573546
  H    2.593603955  -3.027992815  -7.710217276
  C    0.440715499  -3.205589376  -8.060515218
  H    0.340789997  -2.210662643  -8.552334052
  O    0.677423237  -4.159678905  -9.157323978
  C   -0.457601941  -4.992807465  -9.364423296
  H   -0.588356402  -5.099599027 -10.466945566
  C   -0.899487102  -3.648418506  -7.423163303
  H   -0.738236017  -4.281506173  -6.521972445
  C   -1.612903463  -4.427039988  -8.540974718
  H   -2.259402562  -3.745515077  -9.135562231
  H   -2.310698073  -5.195713568  -8.141185547
  O   -1.659525540  -2.468336708  -7.145907268
  N   -0.083943404  -6.357185368  -8.809662357
  C   -0.919375030  -7.494658194  -8.800244510
  C   -0.342296452  -8.415979814  -7.876702638
  N    0.809280215  -7.828710826  -7.341227203
  C    0.943190868  -6.590514335  -7.875023670
  N   -2.059408131  -7.721804761  -9.522348373
  C   -2.681952142  -8.912529417  -9.245946682
  N   -2.198772246  -9.859123590  -8.304726174
  C   -0.994670843  -9.650339112  -7.520649149
  N   -3.862557517  -9.172275555  -9.898066108
  H   -4.183564805  -8.524711705 -10.616998431
  H   -4.320244913 -10.071018359  -9.861417178
  O   -0.682349176 -10.477630522  -6.694287558
  H    1.734209050  -5.863001239  -7.635488524
  H   -2.730242583 -10.718001836  -8.104263774
  P   -2.059346512  -2.294849612  -5.539959032
  O   -2.673790494  -0.769751063  -5.685082092
  C   -1.726615292   0.305849721  -5.578339909
  H   -2.124665319   1.052944122  -6.297823745
  H   -0.699743878   0.021685240  -5.883953781
  C   -1.786735759   0.823306871  -4.130949399
  H   -1.013539836   0.357797597  -3.465230077
  O   -1.386424515   2.229852791  -4.158163010
  C   -2.420761478   3.081407256  -3.653054211
  H   -1.928279499   3.697027133  -2.855074922
  C   -3.222178770   0.762406189  -3.544400232
  H   -3.955011310   0.274747524  -4.237830409
  C   -3.593834089   2.216901354  -3.203155761
  H   -3.758346243   2.292231930  -2.097594031
  H   -4.565988945   2.492786050  -3.652279047
  O   -3.274889545   0.094070446  -2.277897521
  O   -0.839215247  -2.519990831  -4.717041303
  O   -3.351069917  -3.271954028  -5.541299301
  N   -2.783168348   4.001141099  -4.787872991
  C   -2.023339114   4.260419825  -5.921495227
  C   -2.694910379   5.299947332  -6.645355435
  N   -3.857964702   5.670649729  -5.956883780
  C   -3.905339334   4.910203370  -4.863112867
  N   -0.831171329   3.705195519  -6.386044090
  C   -0.332097607   4.232350627  -7.590338146
  N   -0.912808089   5.195777355  -8.287796339
  C   -2.112379331   5.801819696  -7.852973025
  H    0.615916610   3.796992127  -7.977883513
  N   -2.618894606   6.808305361  -8.581845451
  H   -2.154557076   7.142149936  -9.427147211
  H   -3.483850420   7.275462339  -8.304413329
  H   -4.668587537   4.945293853  -4.089383106
  P   -2.827461960  -1.501655127  -2.190966591
  O   -1.319640081  -1.109091767  -1.701029383
  C   -0.324184903  -2.154992170  -1.558117331
  H   -0.184713534  -2.684285921  -2.532621556
  H    0.599303517  -1.587346113  -1.306021378
  C   -0.839581976  -3.034960717  -0.422493339
  H   -1.003859563  -2.469281013   0.527921590
  O   -2.187392098  -3.383197483  -0.900626943
  C   -2.315262298  -4.819963684  -1.081033878
  H   -3.352517500  -5.015849748  -0.709854565
  C   -0.113511704  -4.362912189  -0.138632321
  H    0.743557258  -4.522448426  -0.852243647
  C   -1.197700110  -5.450599723  -0.253973538
  H   -0.794606260  -6.395403386  -0.676809462
  H   -1.566532310  -5.742503702   0.755482528
  O    0.359705260  -4.249156015   1.214321809
  O   -3.001457655  -2.227165353  -3.487808121
  O   -3.904275115  -1.917402676  -1.055555432
  N   -2.242008396  -5.170341805  -2.530364328
  C   -0.999439147  -5.313884135  -3.228468309
  N   -1.069136944  -5.871260305  -4.525227791
  C   -2.281581423  -6.206350701  -5.211882421
  C   -3.513468352  -5.955199137  -4.448244517
  C   -3.463411997  -5.454057886  -3.189158811
  O    0.080070921  -5.029858691  -2.736140165
  H   -0.159516747  -6.089972422  -4.981240843
  O   -2.170412717  -6.646667800  -6.340946600
  C   -4.800194647  -6.278584869  -5.124131658
  H   -5.283432848  -7.161620070  -4.676082840
  H   -5.518247157  -5.445629933  -5.080077966
  H   -4.652327408  -6.503432025  -6.195156881
  H   -4.377202226  -5.250724908  -2.610592495
  P    1.544767563  -5.345310692   1.603025493
  O    2.855275155  -4.679304965   0.872458039
  C    3.487564471  -5.506005125  -0.139047182
  H    4.326003937  -6.039318538   0.358343389
  H    2.798757175  -6.250838060  -0.587411186
  C    4.013651487  -4.487213478  -1.163679844
  H    4.799400030  -3.836094845  -0.715716816
  O    4.713542467  -5.254918493  -2.187652119
  C    4.010279115  -5.223357204  -3.433036594
  H    4.807618306  -5.136509192  -4.209441916
  C    2.909043856  -3.679292093  -1.891208767
  H    1.892085741  -3.882573547  -1.466017802
  C    2.999530541  -4.083025381  -3.369701419
  H    1.989127157  -4.356402458  -3.762608600
  H    3.292298383  -3.233664107  -4.019800534
  O    3.029781007  -2.283893962  -1.696506414
  O    1.243611303  -6.737818793   1.227382608
  O    1.771765986  -4.939967268   3.154876125
  N    3.348745198  -6.582149868  -3.596682411
  C    2.346531094  -6.776599560  -4.618293585
  N    1.779057834  -8.029639781  -4.768098091
  C    2.150620529  -9.081464684  -3.964614666
  C    3.168284234  -8.900169329  -2.962438526
  C    3.748760004  -7.666887323  -2.817211554
  O    2.016026953  -5.849738584  -5.357153306
  N    1.523903324 -10.267162291  -4.207932718
  H    1.730239185 -11.100569344  -3.676283237
  H    0.803372889 -10.343449309  -4.928923296
  H    0.876839290  -2.410581576  -5.493314151
  H    4.560788183  -7.480725025  -2.090302962
  H    3.482172780  -9.738655974  -2.339383221
  H    3.890594750  -1.931900068  -2.016665976
  H   -3.840744163  -3.358712817  -4.643888877
  H   -3.600676539  -1.986337372  -0.102298702
  H    1.468672725  -5.594730638   3.853053768
  H    1.345494068  -8.225296809  -6.531116900
  H   -0.352302189   2.988495196  -5.816096011

