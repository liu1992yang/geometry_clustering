%mem=64gb
%nproc=28       
%Chk=snap_185.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_185 

2     1 
  O    0.238203851  -1.785701160  -5.840066141
  C    0.408379263  -2.821657557  -6.802015158
  H    0.387353913  -3.809632736  -6.298385245
  H    1.434805809  -2.647076004  -7.191801820
  C   -0.642572866  -2.695852738  -7.914259664
  H   -0.734870421  -1.651514571  -8.284880742
  O   -0.123083389  -3.433568950  -9.072802891
  C   -0.958412842  -4.538446035  -9.389031235
  H   -0.910865541  -4.669681759 -10.497184704
  C   -2.022703938  -3.327182134  -7.586447945
  H   -2.020506932  -3.874767881  -6.614689278
  C   -2.327027619  -4.256719121  -8.770297158
  H   -2.993738265  -3.736692936  -9.500548883
  H   -2.903288138  -5.160002683  -8.486021957
  O   -3.029333443  -2.279613857  -7.613932562
  N   -0.277712366  -5.736387907  -8.746816275
  C   -0.835283336  -6.984394295  -8.405770024
  C    0.167334835  -7.685783600  -7.670548936
  N    1.307604202  -6.874616957  -7.588930073
  C    1.034030207  -5.706529821  -8.214118547
  N   -2.090592249  -7.458129266  -8.667038436
  C   -2.357159490  -8.702226263  -8.135001056
  N   -1.426789743  -9.439154353  -7.365430097
  C   -0.089754837  -8.971021945  -7.076875875
  N   -3.601020581  -9.232436754  -8.336963808
  H   -4.265465757  -8.756855721  -8.942531181
  H   -3.876340729 -10.126944779  -7.946935149
  O    0.597976220  -9.641252250  -6.336549373
  H    1.712452106  -4.848831345  -8.318138966
  H   -1.740273387 -10.292228388  -6.835669564
  P   -3.514398253  -1.779990422  -6.121543752
  O   -3.794470841  -0.172419308  -6.272146856
  C   -2.666574621   0.704900433  -6.449676297
  H   -3.166577223   1.699101793  -6.457013824
  H   -2.210436222   0.528143686  -7.446082135
  C   -1.631684482   0.582645629  -5.325159490
  H   -0.961712302  -0.320171706  -5.467479620
  O   -0.730304004   1.727179001  -5.475925312
  C   -0.238933469   2.080360206  -4.157121813
  H    0.713397979   1.503749616  -3.990757500
  C   -2.184614306   0.648057767  -3.875698113
  H   -3.285598138   0.800566343  -3.825091741
  C   -1.372672169   1.753130529  -3.178322282
  H   -0.983226811   1.382621417  -2.198674814
  H   -2.000846293   2.631569272  -2.933052734
  O   -1.780475806  -0.552730962  -3.183372390
  O   -2.331142867  -2.103776110  -5.207457909
  O   -4.965678206  -2.448902371  -6.006700571
  N    0.090033092   3.528640538  -4.219656291
  C    0.028671428   4.382112066  -5.312457396
  C    0.568930183   5.639750509  -4.884906350
  N    0.957832737   5.549588655  -3.540592785
  C    0.682696995   4.307348499  -3.150283163
  N   -0.435093485   4.191460746  -6.612415139
  C   -0.336883883   5.301771850  -7.471492784
  N    0.154020177   6.479273069  -7.122446160
  C    0.645382980   6.721612668  -5.819983646
  H   -0.708424131   5.171361828  -8.512244702
  N    1.152071811   7.932948542  -5.546382743
  H    1.186316414   8.669501470  -6.252278600
  H    1.522694212   8.153720938  -4.620794628
  H    0.856683368   3.875779793  -2.166264479
  P   -2.834747442  -1.814520035  -3.086343296
  O   -1.730138889  -3.010759636  -3.107811678
  C   -2.233455568  -4.352232596  -2.881826405
  H   -3.282676760  -4.487501765  -3.208463338
  H   -1.580554350  -4.965978094  -3.543037341
  C   -2.071431613  -4.696782012  -1.394339790
  H   -2.403215054  -3.865294228  -0.718704876
  O   -3.005168330  -5.785257715  -1.151869074
  C   -2.320394983  -6.985288134  -0.770644459
  H   -2.953479721  -7.409859565   0.048508487
  C   -0.667793759  -5.208348378  -0.980731126
  H    0.055636872  -5.229383846  -1.840315098
  C   -0.903304973  -6.593493496  -0.352723078
  H   -0.124694216  -7.321853484  -0.665267297
  H   -0.833032160  -6.556631576   0.757083409
  O   -0.221782759  -4.240781747  -0.011239330
  O   -4.119442588  -1.691549347  -3.876531886
  O   -3.216548072  -1.840335081  -1.495332558
  N   -2.365691878  -7.935991384  -1.938885028
  C   -1.435412056  -7.856042311  -3.013816478
  N   -1.580612940  -8.826147750  -4.035938014
  C   -2.656044719  -9.772105383  -4.136993262
  C   -3.582210483  -9.760592066  -2.999788403
  C   -3.419562931  -8.880959452  -1.978546998
  O   -0.553241895  -7.016361547  -3.092051253
  H   -0.833712808  -8.854249327  -4.744655744
  O   -2.706916822 -10.446076207  -5.152265605
  C   -4.696692490 -10.747484916  -3.030134892
  H   -4.896894031 -11.192553105  -2.043443396
  H   -5.634957312 -10.288322801  -3.381353217
  H   -4.480612164 -11.591163499  -3.710132168
  H   -4.111013166  -8.861151814  -1.121828189
  P    1.315940960  -4.424358417   0.571582046
  O    1.713624064  -2.835732892   0.634960162
  C    2.923736901  -2.414738159  -0.036400548
  H    3.198443599  -1.523306006   0.572968808
  H    3.732132909  -3.173991764   0.043831455
  C    2.686850078  -2.031048146  -1.501974895
  H    3.191765661  -1.061048576  -1.743163615
  O    3.402888233  -3.020241577  -2.306012964
  C    2.591828495  -3.435436029  -3.424076722
  H    2.992520979  -2.915892244  -4.331037005
  C    1.222228556  -2.003958263  -1.991746573
  H    0.463993987  -2.111591396  -1.181888622
  C    1.139526955  -3.084553739  -3.074585445
  H    0.569699981  -3.972997958  -2.730399555
  H    0.583233659  -2.729591702  -3.971306541
  O    1.034065362  -0.689350912  -2.557260061
  O    2.179981305  -5.367484806  -0.148205030
  O    1.019274259  -4.756622089   2.134003784
  N    2.784218506  -4.910111711  -3.604984691
  C    2.457562620  -5.463562667  -4.906819312
  N    2.470872768  -6.834115328  -5.075261039
  C    2.794894768  -7.658831147  -4.022266954
  C    3.241513030  -7.111034216  -2.772353513
  C    3.220428709  -5.752272189  -2.583132360
  O    2.154539885  -4.700611861  -5.822389803
  N    2.643674916  -8.999026545  -4.238691034
  H    2.916607045  -9.685129912  -3.552672872
  H    2.282605307  -9.345953504  -5.126352863
  H   -0.567447054  -1.967549567  -5.261975348
  H    3.543933989  -5.270995265  -1.637999942
  H    3.581946183  -7.761652862  -1.966864006
  H    0.233192565  -0.720289233  -3.130030358
  H   -5.452964080  -2.296519743  -5.103665004
  H   -4.113184511  -1.524093683  -1.208250567
  H    0.646952381  -4.031635267   2.712252571
  H    2.093245673  -7.063541303  -6.903193791
  H   -0.768244987   3.257206762  -6.894520650

