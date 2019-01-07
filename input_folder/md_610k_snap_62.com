%mem=64gb
%nproc=28       
%Chk=snap_62.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_62 

2     1 
  O    1.147248457  -6.662587709  -6.916637842
  C    2.147662461  -5.676407792  -7.193604189
  H    2.546755739  -5.972703717  -8.186909277
  H    2.944204481  -5.731487371  -6.427698895
  C    1.549490642  -4.262448505  -7.270125190
  H    2.319587041  -3.469510378  -7.121715192
  O    1.112541529  -4.044988612  -8.652491889
  C   -0.298331173  -4.068879445  -8.746988545
  H   -0.551068108  -3.368752210  -9.583932285
  C    0.306344960  -4.031690587  -6.381239203
  H    0.111292356  -4.901790115  -5.708988113
  C   -0.851128760  -3.737772839  -7.362049860
  H   -1.144532058  -2.664537657  -7.318781433
  H   -1.774576998  -4.285609985  -7.085928969
  O    0.632310981  -2.888118559  -5.555291936
  N   -0.715043319  -5.451401158  -9.202004158
  C   -0.765749613  -5.867824695 -10.554283690
  C   -0.998071436  -7.271389309 -10.546307291
  N   -1.061847075  -7.689249890  -9.202347781
  C   -0.881675624  -6.601749623  -8.408611169
  N   -0.661727458  -5.087071998 -11.670127525
  C   -0.780356235  -5.769512341 -12.860414608
  N   -0.995561251  -7.161883796 -12.940742139
  C   -1.122265202  -8.028131031 -11.769074816
  N   -0.700384555  -5.036842011 -14.016353308
  H   -0.514568656  -4.038253392 -13.964811068
  H   -0.723885517  -5.454494021 -14.935419383
  O   -1.310338818  -9.201055551 -11.954327425
  H   -0.873113849  -6.603281082  -7.294733323
  H   -1.081062184  -7.631345026 -13.854953507
  P   -0.495562457  -2.443792303  -4.426839734
  O   -1.260616061  -1.217463352  -5.254587631
  C   -0.912098455   0.125676390  -4.906947980
  H   -1.122592654   0.675203961  -5.848500963
  H    0.162121094   0.236785175  -4.659655323
  C   -1.810176967   0.622842878  -3.757153161
  H   -1.349677504   0.478513264  -2.749877390
  O   -1.869629914   2.082428242  -3.867161703
  C   -3.158633716   2.519019211  -4.299220373
  H   -3.382597051   3.428276227  -3.682780376
  C   -3.270664739   0.109026957  -3.858518541
  H   -3.385976789  -0.703012696  -4.619093956
  C   -4.129670120   1.352834058  -4.159673924
  H   -4.827275295   1.517264701  -3.295097165
  H   -4.793602752   1.185103928  -5.024506282
  O   -3.781801022  -0.337027188  -2.592244437
  O    0.167740742  -2.050368911  -3.166345703
  O   -1.520859965  -3.673644237  -4.699798226
  N   -2.971734249   2.959406053  -5.729982201
  C   -1.789856348   3.458036468  -6.268154048
  C   -2.060134302   3.789211664  -7.635514939
  N   -3.398245424   3.494620546  -7.929709969
  C   -3.931328445   3.009826089  -6.806765079
  N   -0.527361439   3.651713202  -5.707784058
  C    0.445579731   4.210137374  -6.555412231
  N    0.241649331   4.530536364  -7.823915289
  C   -1.016207863   4.351185901  -8.438592223
  H    1.455575071   4.388736912  -6.120956962
  N   -1.156145936   4.710334594  -9.724382636
  H   -0.384776439   5.121162533 -10.251502994
  H   -2.054093557   4.615587615 -10.200981482
  H   -4.964748768   2.699341927  -6.673509101
  P   -3.145411142  -1.733940210  -1.971321494
  O   -2.347550978  -1.048585276  -0.737399589
  C   -1.186237050  -1.803658582  -0.277422130
  H   -0.487044862  -1.995945674  -1.121924128
  H   -0.707193540  -1.092583062   0.435656293
  C   -1.724219323  -3.051891377   0.407339958
  H   -2.213635660  -2.823398240   1.384043245
  O   -2.814537735  -3.513856947  -0.475437017
  C   -2.827865775  -4.982012454  -0.543071154
  H   -3.872309968  -5.240627590  -0.237636312
  C   -0.769140096  -4.265279006   0.511657510
  H   -0.006611457  -4.277388384  -0.312975319
  C   -1.743381082  -5.458216268   0.418145145
  H   -1.230027968  -6.391710840   0.112186898
  H   -2.152041422  -5.688387011   1.429360500
  O   -0.172139387  -4.350494471   1.796930053
  O   -2.343209844  -2.425453359  -3.030671398
  O   -4.542393505  -2.442287942  -1.543210150
  N   -2.629510661  -5.390271845  -1.962689723
  C   -1.355507086  -5.817612132  -2.470203288
  N   -1.344828921  -6.319842650  -3.788838995
  C   -2.440833498  -6.265760879  -4.702051121
  C   -3.687717989  -5.732109774  -4.124272747
  C   -3.750882456  -5.334519263  -2.828186048
  O   -0.333352613  -5.783981981  -1.806616671
  H   -0.423308372  -6.680637031  -4.125286275
  O   -2.240635367  -6.628807805  -5.848606113
  C   -4.854488792  -5.646067129  -5.043676918
  H   -4.579722486  -5.155282095  -5.991335347
  H   -5.228930078  -6.651973948  -5.303325281
  H   -5.701162252  -5.087800317  -4.621237087
  H   -4.675249083  -4.924723369  -2.389461250
  P    1.188956965  -3.440865013   2.113723620
  O    2.465183281  -4.415041647   1.782505198
  C    2.524869715  -5.269187162   0.632759841
  H    3.184689467  -6.096249517   0.992124498
  H    1.548372794  -5.704831412   0.362504261
  C    3.223180557  -4.497731314  -0.494196967
  H    3.920267127  -3.725165024  -0.078580136
  O    4.108837496  -5.440808694  -1.169222908
  C    3.762598982  -5.568409414  -2.555219625
  H    4.734746172  -5.694311405  -3.088440751
  C    2.299390995  -3.911769009  -1.589273004
  H    1.233296424  -4.245271459  -1.512847592
  C    2.953540601  -4.325086940  -2.913949494
  H    2.215394213  -4.474835656  -3.732075633
  H    3.623426816  -3.517806121  -3.280570602
  O    2.352087219  -2.486136078  -1.412005535
  O    1.220410233  -2.875928698   3.447985982
  O    1.066138857  -2.426418276   0.830524372
  N    3.000479805  -6.876434504  -2.655493444
  C    1.926393552  -7.109553219  -3.591983285
  N    1.300914920  -8.337038479  -3.638313507
  C    1.680347226  -9.331131626  -2.773973633
  C    2.730480832  -9.115483204  -1.808441183
  C    3.360948616  -7.902369133  -1.776266145
  O    1.574691966  -6.200893943  -4.358095794
  N    1.012809729 -10.520329135  -2.863376950
  H    1.250861354 -11.311178811  -2.284780161
  H    0.308257983 -10.664781863  -3.576075353
  H    1.092993287  -6.828904370  -5.925241145
  H    4.174911292  -7.682117553  -1.060881962
  H    3.021980664  -9.910456107  -1.121957672
  H    1.710555239  -2.053062188  -2.063271786
  H   -2.305592638  -3.753468279  -4.017461104
  H   -4.716021782  -2.552628603  -0.553862033
  H    1.850451657  -2.256931851   0.187445373
  H   -1.207426674  -8.652440441  -8.914530787
  H   -0.363191994   3.389325918  -4.722329918

