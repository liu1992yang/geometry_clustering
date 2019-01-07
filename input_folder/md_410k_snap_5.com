%mem=64gb
%nproc=28       
%Chk=snap_5.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_5 

2     1 
  O    1.944165603  -3.724870828  -7.672079178
  C    2.061960892  -2.629006806  -8.589648983
  H    2.764658736  -3.011333517  -9.364156938
  H    2.518862522  -1.764702451  -8.068052987
  C    0.718788459  -2.253980064  -9.227559247
  H    0.684195848  -1.173555033  -9.509103757
  O    0.633929543  -2.954130871 -10.515117790
  C   -0.548190360  -3.733918064 -10.595121490
  H   -0.886241817  -3.681517954 -11.660618406
  C   -0.551370910  -2.675567605  -8.445129929
  H   -0.339446995  -3.476367667  -7.679345116
  C   -1.506642367  -3.218061507  -9.515861918
  H   -2.161078777  -2.405877149  -9.903513354
  H   -2.200693130  -3.984048575  -9.119660188
  O   -1.142839341  -1.488728135  -7.859324843
  N   -0.154097572  -5.165390696 -10.306952737
  C   -0.430868670  -6.276211411 -11.134031247
  C    0.090034650  -7.421500033 -10.469024254
  N    0.684976211  -6.988265529  -9.267320921
  C    0.541246218  -5.641896450  -9.181305853
  N   -1.071149555  -6.272672651 -12.340029843
  C   -1.173893434  -7.510389528 -12.936232309
  N   -0.701036360  -8.699398567 -12.339122361
  C   -0.025389574  -8.741273349 -11.042303638
  N   -1.775848556  -7.567681770 -14.164902938
  H   -2.099531740  -6.709778517 -14.607000295
  H   -1.872893202  -8.422495648 -14.693152658
  O    0.331655438  -9.812646071 -10.629082282
  H    0.931558831  -4.975370488  -8.340203845
  H   -0.813643479  -9.608919525 -12.812466393
  P   -0.814228596  -1.419328449  -6.249461900
  O   -1.180886859   0.164822252  -5.980109637
  C   -0.649001142   0.722210580  -4.752252113
  H   -0.207452913   1.691408421  -5.089225863
  H    0.159836798   0.099389967  -4.312546873
  C   -1.772191396   0.990858139  -3.745810011
  H   -1.368672064   0.981947811  -2.692633360
  O   -2.170110988   2.390675599  -3.901244214
  C   -3.520992415   2.507229692  -4.321465155
  H   -3.948742090   3.337164948  -3.711931434
  C   -3.062006570   0.139985158  -3.877955489
  H   -3.027906849  -0.650483636  -4.669517428
  C   -4.196629377   1.148554729  -4.134739230
  H   -4.873470777   1.155142758  -3.241523411
  H   -4.850186956   0.843764425  -4.972910284
  O   -3.407272827  -0.453569106  -2.612767567
  O    0.538768957  -1.888261188  -5.856714362
  O   -2.065098431  -2.166789321  -5.589569585
  N   -3.488471845   2.939341903  -5.763292612
  C   -3.649301126   4.231678154  -6.261808857
  C   -3.356554533   4.174979433  -7.664216277
  N   -2.992816586   2.865658694  -8.015991509
  C   -3.057515127   2.148930913  -6.895423147
  N   -4.041805953   5.431404132  -5.665586016
  C   -4.121785453   6.555712421  -6.514020533
  N   -3.852959867   6.543464711  -7.807416489
  C   -3.447481423   5.359413999  -8.460542933
  H   -4.439211336   7.515811091  -6.046877547
  N   -3.170005540   5.420495844  -9.773089387
  H   -3.249419717   6.293655708 -10.294677385
  H   -2.873513555   4.590014720 -10.285522163
  H   -2.809306173   1.088597137  -6.787169459
  P   -2.533176683  -1.757300696  -2.095384066
  O   -1.285194217  -0.851373203  -1.533685956
  C   -0.130076850  -1.501592308  -0.963943034
  H    0.537518801  -1.797123932  -1.808535739
  H    0.349230219  -0.684237502  -0.383053170
  C   -0.535348636  -2.674881163  -0.071937610
  H   -1.065045659  -2.360142481   0.863626853
  O   -1.544944381  -3.424225279  -0.824148098
  C   -1.076125235  -4.780638675  -1.150950622
  H   -1.956022278  -5.410861115  -0.863937640
  C    0.615986550  -3.656015629   0.244982596
  H    1.608094995  -3.324269857  -0.149355337
  C    0.174496027  -5.014539949  -0.310624164
  H    1.002051331  -5.501077136  -0.891680170
  H   -0.035409670  -5.739760748   0.513287900
  O    0.632605833  -3.660997213   1.693066512
  O   -2.206791818  -2.761567232  -3.155209315
  O   -3.552429281  -2.163833192  -0.908812203
  N   -0.850254173  -4.892071571  -2.612953450
  C    0.226883555  -4.208149536  -3.252076264
  N    0.301276450  -4.345534135  -4.658389512
  C   -0.562152320  -5.162606645  -5.446871609
  C   -1.600493214  -5.865422476  -4.709351203
  C   -1.725074645  -5.709078298  -3.362221212
  O    1.039953258  -3.528624906  -2.649088842
  H    1.016168823  -3.759339467  -5.138467716
  O   -0.359561100  -5.158403932  -6.660335580
  C   -2.508877968  -6.761909058  -5.480137830
  H   -2.274696088  -6.751492981  -6.555986852
  H   -2.421053664  -7.807752196  -5.143413856
  H   -3.561427370  -6.460498835  -5.376714379
  H   -2.502675874  -6.235772562  -2.787923151
  P    1.907149004  -4.380345761   2.434149415
  O    1.765626844  -5.885070049   1.777233414
  C    2.554232172  -6.935534336   2.392561624
  H    3.054915514  -6.601628050   3.324335795
  H    1.797268116  -7.706929709   2.658898629
  C    3.556915743  -7.500347750   1.378522214
  H    4.465728575  -7.886667825   1.901820119
  O    2.997684170  -8.710914035   0.788232968
  C    2.512820062  -8.467513598  -0.543409042
  H    2.879262724  -9.335842379  -1.143162389
  C    3.918021239  -6.584918307   0.180028389
  H    3.793767927  -5.492575737   0.377460838
  C    3.044278773  -7.098545675  -0.977994753
  H    2.230281882  -6.368197112  -1.205343789
  H    3.613206581  -7.158056272  -1.926603502
  O    5.304158399  -6.665509379  -0.102865850
  O    2.019107647  -4.227046090   3.877697862
  O    3.163361968  -3.814158670   1.520464066
  N    1.006075275  -8.555523196  -0.466081814
  C    0.155570269  -7.962721786  -1.510273908
  N   -1.218630353  -7.992433311  -1.318500710
  C   -1.753269401  -8.669455848  -0.260496828
  C   -0.932601509  -9.336767292   0.710354415
  C    0.431420807  -9.253361203   0.585721424
  O    0.707464426  -7.437735859  -2.459842567
  N   -3.125847350  -8.628013731  -0.130909720
  H   -3.594621939  -9.225966662   0.533886991
  H   -3.691758038  -8.282418656  -0.893493506
  H    1.574241120  -3.401973393  -6.797518824
  H    1.115076845  -9.735852386   1.307657271
  H   -1.379431905  -9.902251828   1.525181184
  H    5.588419560  -7.573257318  -0.349572700
  H   -1.931694502  -2.653612041  -4.623449058
  H   -3.354708771  -3.023598072  -0.404389105
  H    3.901162561  -3.327992650   1.973299778
  H    1.125787350  -7.609746160  -8.594387301
  H   -4.229117883   5.497430855  -4.661919386

