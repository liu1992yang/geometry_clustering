%mem=64gb
%nproc=28       
%Chk=snap_88.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_88 

2     1 
  O    2.794511929  -4.745910542  -7.512042953
  C    2.493509291  -3.546082703  -8.235243230
  H    3.043496931  -3.703798938  -9.193428048
  H    2.924835420  -2.681969716  -7.695501292
  C    1.006506235  -3.327709997  -8.514577518
  H    0.845721370  -2.315549615  -8.967228411
  O    0.611547641  -4.262024870  -9.578396685
  C   -0.596602028  -4.931324035  -9.247600283
  H   -1.186510037  -5.021083173 -10.190281652
  C   -0.000827419  -3.546378919  -7.356206726
  H    0.406154568  -4.177898232  -6.529974234
  C   -1.214850038  -4.182160065  -8.063810782
  H   -1.923702705  -3.397626181  -8.406757563
  H   -1.831473766  -4.819581520  -7.393658205
  O   -0.297833899  -2.225656022  -6.881328827
  N   -0.225821604  -6.329246733  -8.794307501
  C   -1.137702763  -7.398493103  -8.640319883
  C   -0.550547782  -8.302146694  -7.705808257
  N    0.707236783  -7.789609963  -7.340453320
  C    0.883771233  -6.609751452  -7.974203672
  N   -2.334115676  -7.594552948  -9.272321683
  C   -3.001862347  -8.729280100  -8.889760405
  N   -2.518455280  -9.640258514  -7.914980062
  C   -1.258141233  -9.453725855  -7.220035533
  N   -4.224613367  -8.974972808  -9.469086195
  H   -4.548321234  -8.363012412 -10.216620222
  H   -4.700375377  -9.859747752  -9.371933892
  O   -0.960583421 -10.238111149  -6.344133411
  H    1.809366985  -5.911511066  -7.909111668
  H   -3.090053556 -10.441764418  -7.615567423
  P   -1.002501117  -2.129695521  -5.376098177
  O   -1.821585940  -0.717880695  -5.677213526
  C   -1.155126393   0.507072106  -5.344910652
  H   -1.477550185   1.194550462  -6.156448579
  H   -0.051340176   0.422071378  -5.360174253
  C   -1.676099303   0.980955866  -3.975688586
  H   -1.007259322   0.687682139  -3.128125788
  O   -1.587081424   2.444325469  -3.971860515
  C   -2.856267992   3.046416509  -3.690976563
  H   -2.691069066   3.684125090  -2.782687196
  C   -3.162720906   0.611372691  -3.735659913
  H   -3.586749245  -0.023147252  -4.555689721
  C   -3.905018988   1.946257303  -3.554684141
  H   -4.380605907   1.954836100  -2.540899014
  H   -4.748656717   2.028366991  -4.265557875
  O   -3.359573537  -0.043117313  -2.473231645
  O    0.009996652  -2.128529745  -4.300816298
  O   -2.123709369  -3.287187669  -5.601080390
  N   -3.149483403   3.964322962  -4.844857252
  C   -2.251862325   4.463349386  -5.778777571
  C   -2.982122650   5.373216035  -6.612884853
  N   -4.314969460   5.432618422  -6.184414044
  C   -4.410227989   4.609314005  -5.141135759
  N   -0.892071726   4.228026814  -5.979106747
  C   -0.289875482   4.942164794  -7.031358490
  N   -0.924532256   5.787245686  -7.827502651
  C   -2.300297105   6.064028346  -7.665068557
  H    0.796867640   4.772919413  -7.201119874
  N   -2.876873986   6.949990720  -8.491354249
  H   -2.346306503   7.425202615  -9.222242254
  H   -3.867445968   7.183651287  -8.403154722
  H   -5.300850721   4.425812385  -4.544593663
  P   -2.746659487  -1.569120204  -2.279520295
  O   -1.489103763  -1.065534782  -1.368162376
  C   -0.373562956  -1.990947683  -1.260554097
  H   -0.035758855  -2.307679941  -2.278814599
  H    0.415338152  -1.365045770  -0.791929409
  C   -0.872383952  -3.114656383  -0.359497472
  H   -1.069388025  -2.766178168   0.683147610
  O   -2.196744231  -3.420328960  -0.927144729
  C   -2.386406983  -4.865854556  -1.068359703
  H   -3.394678110  -5.029939352  -0.613172102
  C   -0.102262289  -4.450033599  -0.361139469
  H    0.551895518  -4.570234070  -1.274337201
  C   -1.225651270  -5.506107012  -0.310939521
  H   -0.900397840  -6.489101491  -0.722878467
  H   -1.517364603  -5.743083432   0.736142444
  O    0.732951516  -4.406765401   0.802562415
  O   -2.449640331  -2.229904338  -3.586883340
  O   -4.011303254  -2.170480240  -1.455659361
  N   -2.453134561  -5.204707600  -2.517027125
  C   -1.295646096  -5.615862800  -3.254516490
  N   -1.506767435  -6.088178178  -4.564794473
  C   -2.774806858  -6.124206466  -5.230057911
  C   -3.893033824  -5.580616758  -4.436535484
  C   -3.715439533  -5.152822823  -3.161709267
  O   -0.174774642  -5.601249736  -2.775094301
  H   -0.653402240  -6.415820071  -5.059582474
  O   -2.798135140  -6.572218858  -6.358142626
  C   -5.208859280  -5.511585145  -5.129615493
  H   -5.314959761  -4.567160384  -5.689371674
  H   -5.319397345  -6.325108196  -5.869099408
  H   -6.064712098  -5.589760458  -4.444208144
  H   -4.540728452  -4.733231799  -2.566181404
  P    1.659889166  -5.764316693   1.068631060
  O    2.828671440  -5.680295779  -0.067249474
  C    3.995098341  -4.840408656   0.146748389
  H    3.923057928  -4.224283769   1.067571634
  H    4.832429713  -5.566521315   0.255868121
  C    4.185791412  -3.968952587  -1.098288939
  H    5.123131476  -3.365991583  -0.992383698
  O    4.466434307  -4.807739337  -2.254189813
  C    3.411161577  -4.738679125  -3.222555342
  H    3.920572313  -4.488585200  -4.184768120
  C    2.965625996  -3.086355570  -1.469065901
  H    2.247488559  -2.981617877  -0.624845767
  C    2.377257391  -3.723466176  -2.732192021
  H    1.394624952  -4.221096170  -2.521029012
  H    2.102293090  -2.975910430  -3.509774144
  O    3.355896063  -1.736369886  -1.670481958
  O    0.920030454  -7.024486139   1.068528179
  O    2.438589849  -5.228789914   2.398793570
  N    2.825019847  -6.120469556  -3.372421877
  C    1.922471038  -6.361793391  -4.476659681
  N    1.396238989  -7.624712790  -4.651240495
  C    1.602134565  -8.602095256  -3.700241813
  C    2.434994101  -8.347993586  -2.558929106
  C    3.033701891  -7.120669285  -2.426230465
  O    1.654607766  -5.436266808  -5.248903842
  N    0.988093142  -9.797153341  -3.932586250
  H    1.077340214 -10.574457431  -3.295163937
  H    0.390008133  -9.935977822  -4.750194543
  H    2.546387057  -4.673655136  -6.538284847
  H    3.710920822  -6.879656966  -1.586322277
  H    2.586036828  -9.116348622  -1.800267167
  H    3.966334854  -1.634664245  -2.433445180
  H   -2.729512392  -3.473749000  -4.790722323
  H   -3.914174584  -2.259953084  -0.456889143
  H    2.078091935  -5.461606836   3.298817424
  H    1.259282893  -8.147028823  -6.544460656
  H   -0.391559083   3.591217555  -5.340351585

