%mem=64gb
%nproc=28       
%Chk=snap_183.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_183 

2     1 
  O    3.078271276  -4.313409821  -4.604163943
  C    3.259168305  -3.071565461  -5.298713719
  H    4.265173646  -3.189527510  -5.759907860
  H    3.293193377  -2.260503127  -4.543473954
  C    2.196955347  -2.786223909  -6.363328312
  H    2.241886087  -1.721268678  -6.707409294
  O    2.512815018  -3.531067571  -7.577157869
  C    1.514230533  -4.497268376  -7.871264237
  H    1.313160337  -4.403456297  -8.969937257
  C    0.761297012  -3.185118960  -5.954483218
  H    0.706748052  -3.532567608  -4.890583343
  C    0.323008629  -4.278151244  -6.938250829
  H   -0.583062896  -3.962723886  -7.501049779
  H    0.004524866  -5.191821779  -6.387604004
  O   -0.029874158  -1.993823475  -6.179430522
  N    2.146238233  -5.847263109  -7.647853691
  C    2.166923517  -6.895476613  -8.584443848
  C    2.854168555  -7.982750740  -7.968588187
  N    3.254157954  -7.564376135  -6.682349290
  C    2.829957466  -6.293484593  -6.500102240
  N    1.636750750  -6.895281449  -9.846450976
  C    1.819509561  -8.059666584 -10.546358125
  N    2.488955245  -9.188864506 -10.011436455
  C    3.046931429  -9.222264103  -8.670577613
  N    1.296155216  -8.128126261 -11.811634909
  H    0.841231971  -7.306131559 -12.206135979
  H    1.463957586  -8.902374210 -12.436586905
  O    3.579182513 -10.254858265  -8.322946663
  H    3.002630499  -5.618181369  -5.583970181
  H    2.597717091 -10.049710932 -10.567576790
  P   -1.229779587  -1.770619500  -5.069653544
  O   -2.211598107  -0.700371689  -5.840491131
  C   -1.659878753   0.616695415  -6.067873171
  H   -2.420299414   1.076224607  -6.735499849
  H   -0.704850724   0.549965886  -6.628528708
  C   -1.516135807   1.401173630  -4.755000102
  H   -0.496357174   1.314174787  -4.296731704
  O   -1.632541537   2.807882598  -5.107308368
  C   -2.657743391   3.458636592  -4.350039015
  H   -2.158260431   4.354770206  -3.894235065
  C   -2.618963354   1.100248661  -3.708260134
  H   -3.383000802   0.383719209  -4.092878628
  C   -3.222455518   2.465642686  -3.339160975
  H   -2.889124381   2.723622315  -2.300365584
  H   -4.325032674   2.446921832  -3.256034431
  O   -1.964053439   0.617847451  -2.526194012
  O   -0.583579967  -1.319205188  -3.781697122
  O   -1.954547043  -3.200865166  -5.205827553
  N   -3.642539931   3.981010048  -5.348553514
  C   -4.742263494   3.391696129  -5.967677991
  C   -5.235501608   4.343786895  -6.923614597
  N   -4.424530958   5.488622801  -6.915254747
  C   -3.484619065   5.268029520  -6.001204921
  N   -5.381989195   2.166443349  -5.804475127
  C   -6.558138842   1.969390076  -6.548914595
  N   -7.045609248   2.816822162  -7.442780577
  C   -6.414912668   4.055096873  -7.679400929
  H   -7.100934654   1.014273313  -6.369659518
  N   -6.957403263   4.884440789  -8.587645182
  H   -7.811094276   4.640380208  -9.087739838
  H   -6.535450302   5.791327754  -8.788081830
  H   -2.675165099   5.943339658  -5.729960920
  P   -2.169716528  -0.995450406  -2.210200321
  O   -0.892953589  -1.378676056  -1.278726526
  C   -0.354221069  -2.707526580  -1.450454649
  H    0.158349010  -2.782158669  -2.431094697
  H    0.419594790  -2.736270496  -0.650384666
  C   -1.455041101  -3.749326583  -1.215208758
  H   -1.305084720  -4.284168122  -0.246328442
  O   -2.697600332  -2.989136749  -1.089521467
  C   -3.837990499  -3.881143682  -1.407702154
  H   -4.259334468  -4.165094154  -0.410174071
  C   -1.757927158  -4.759779799  -2.351500503
  H   -1.512762711  -4.396493592  -3.379889673
  C   -3.253494274  -5.064555518  -2.181718185
  H   -3.751186506  -5.260224282  -3.160817698
  H   -3.382428980  -6.021879700  -1.624597326
  O   -1.073915524  -6.005380756  -2.058266046
  O   -2.799586258  -1.714448178  -3.402883488
  O   -3.280107248  -0.744549905  -1.008279025
  N   -4.839427651  -3.023742029  -2.100942990
  C   -4.793183312  -2.848308437  -3.519413197
  N   -5.331049808  -1.643169210  -4.053121691
  C   -5.774421134  -0.559416081  -3.247729055
  C   -5.975805710  -0.873819862  -1.834332251
  C   -5.460628932  -2.015472356  -1.306577873
  O   -4.432054441  -3.732115414  -4.288249037
  H   -5.181512260  -1.496583907  -5.061231243
  O   -5.926505915   0.526082083  -3.805239160
  C   -6.682691876   0.136036798  -0.999930839
  H   -7.557319781  -0.289910999  -0.483789683
  H   -6.014018199   0.559871708  -0.232784265
  H   -7.052769260   0.983895964  -1.599599991
  H   -5.517210467  -2.231674204  -0.230725522
  P    0.451589882  -6.087415235  -2.642223405
  O    1.256058777  -5.278621090  -1.451124516
  C    2.693523566  -5.437203518  -1.427272499
  H    3.123485870  -5.750604796  -2.407410830
  H    3.047214095  -4.404686774  -1.203495438
  C    3.052469588  -6.388933474  -0.285555132
  H    2.430947935  -6.182603177   0.619333389
  O    2.688202180  -7.766514934  -0.612060772
  C    3.859166393  -8.620116258  -0.708192340
  H    3.696611772  -9.377824987   0.101388528
  C    4.579940332  -6.383829406   0.004336809
  H    5.118950380  -5.526574966  -0.467799216
  C    5.092682693  -7.745044763  -0.493876813
  H    5.680430750  -7.613704361  -1.429550064
  H    5.811536121  -8.192604338   0.221581194
  O    4.825729579  -6.158053744   1.376869534
  O    0.678255820  -5.408975162  -3.949692906
  O    0.786485185  -7.653702848  -2.542749775
  N    3.800636245  -9.296394361  -2.033779658
  C    3.921792224  -8.563438260  -3.302531761
  N    3.919205156  -9.293842886  -4.476997816
  C    3.860720002 -10.663684428  -4.467982155
  C    3.745049573 -11.391601162  -3.222428298
  C    3.711629961 -10.696217975  -2.051343265
  O    3.996055578  -7.348065278  -3.275183404
  N    3.908090394 -11.296313870  -5.670141372
  H    3.880945586 -12.302772065  -5.749258275
  H    3.983657667 -10.767191139  -6.545810223
  H    2.146719729  -4.397598238  -4.214303252
  H    3.620270740 -11.210480704  -1.081474788
  H    3.683936084 -12.480488145  -3.229352524
  H    4.502094790  -6.886610751   1.951198054
  H   -2.936013179  -3.277721314  -4.882363224
  H   -2.954143268  -0.909478706  -0.078381328
  H    1.605900997  -7.932323530  -1.954217605
  H    3.725116522  -8.182488818  -5.972514420
  H   -5.127627120   1.490090858  -5.044966562

