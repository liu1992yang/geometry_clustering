%mem=64gb
%nproc=28       
%Chk=snap_20.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_20 

2     1 
  O   -0.103622263  -6.187201695  -7.656108868
  C    0.920821594  -5.281831553  -7.235816854
  H    1.841175589  -5.694322791  -7.708657135
  H    1.030157164  -5.317426545  -6.132763717
  C    0.675219482  -3.850471255  -7.724863910
  H    1.273529535  -3.110358535  -7.134194002
  O    1.261967902  -3.722553221  -9.058929096
  C    0.264430620  -3.586273160 -10.051336589
  H    0.690297887  -2.882068910 -10.813170846
  C   -0.810162334  -3.421160499  -7.853717203
  H   -1.535023158  -4.160472001  -7.438886803
  C   -1.030902092  -3.162843544  -9.358578675
  H   -1.211020361  -2.068652428  -9.504080262
  H   -1.945732008  -3.642053001  -9.746212247
  O   -0.962079637  -2.134450421  -7.234758103
  N    0.136265909  -4.932917537 -10.738754686
  C    0.876415502  -5.336947141 -11.877270389
  C    0.633883038  -6.730603912 -12.050926516
  N   -0.212502001  -7.156297788 -11.013091225
  C   -0.493711614  -6.082548646 -10.230462008
  N    1.656603809  -4.563322234 -12.684792524
  C    2.254792167  -5.238490245 -13.728983359
  N    2.069239632  -6.617491508 -13.968472513
  C    1.232793910  -7.477443333 -13.134164278
  N    3.062171209  -4.517329595 -14.566435364
  H    3.213907480  -3.525518901 -14.393019916
  H    3.561851566  -4.930217172 -15.341897521
  O    1.143354166  -8.639378394 -13.426412451
  H   -1.092881964  -6.089576431  -9.299617289
  H    2.541716085  -7.084585069 -14.758033048
  P   -1.444748270  -2.106101136  -5.642135988
  O   -1.715923941  -0.469316476  -5.644980382
  C   -0.792637605   0.319126848  -4.860470864
  H   -0.319841400   1.018356789  -5.585597001
  H    0.007029407  -0.292010736  -4.386877457
  C   -1.608339796   1.085681496  -3.813728570
  H   -1.114331279   1.067792527  -2.805699863
  O   -1.562055605   2.505324771  -4.174822589
  C   -2.866642652   3.055371620  -4.268828343
  H   -2.812216197   4.040366304  -3.752159969
  C   -3.112906322   0.700281216  -3.735630684
  H   -3.459705247   0.066244143  -4.589474542
  C   -3.855768112   2.045552685  -3.686459718
  H   -4.106989860   2.283120813  -2.624227802
  H   -4.831153306   2.009221503  -4.205456690
  O   -3.409694745   0.035840340  -2.502214330
  O   -0.403852343  -2.641287989  -4.742649302
  O   -2.894510046  -2.797873605  -5.885547571
  N   -3.116967043   3.286103101  -5.737143982
  C   -3.393438945   4.493335228  -6.371867431
  C   -3.427778464   4.222584478  -7.780034404
  N   -3.158088389   2.863628306  -8.002645591
  C   -2.967566313   2.318720584  -6.801254030
  N   -3.639801367   5.778964128  -5.885137828
  C   -3.902787673   6.773325153  -6.850551596
  N   -3.935459076   6.565020363  -8.154525218
  C   -3.700729954   5.284114711  -8.700050073
  H   -4.096923591   7.803348196  -6.472407811
  N   -3.744600432   5.146257698 -10.035023006
  H   -3.939972882   5.938153340 -10.648769929
  H   -3.587951791   4.238924517 -10.473067580
  H   -2.719195796   1.271314941  -6.594315968
  P   -2.744228454  -1.482929836  -2.343499037
  O   -1.330734165  -0.898927742  -1.761194226
  C   -0.226868510  -1.832959734  -1.602116789
  H   -0.035287546  -2.380469269  -2.557908372
  H    0.629985811  -1.168865793  -1.372654861
  C   -0.641283852  -2.721284761  -0.435994963
  H   -0.782380841  -2.159529560   0.519250078
  O   -1.997816216  -3.131509362  -0.833460392
  C   -2.094781411  -4.592449800  -0.915000237
  H   -3.075492173  -4.815620143  -0.427697409
  C    0.172530614  -4.011286613  -0.211836653
  H    0.970692617  -4.164316204  -0.980648019
  C   -0.876038824  -5.130428508  -0.194282928
  H   -0.475389550  -6.095428484  -0.605560351
  H   -1.119515609  -5.427709683   0.863591498
  O    0.726967825  -3.860245990   1.114272098
  O   -2.637776013  -2.189598327  -3.660861525
  O   -3.894878883  -2.036706199  -1.349273855
  N   -2.183261807  -4.947579207  -2.365064131
  C   -1.014294149  -5.150212612  -3.157007558
  N   -1.232385644  -5.544338732  -4.500030703
  C   -2.510630315  -5.642206422  -5.132782510
  C   -3.652538436  -5.370123004  -4.261060887
  C   -3.464307636  -5.027258263  -2.958865706
  O    0.125072924  -5.064634266  -2.733581477
  H   -0.384215623  -5.673240861  -5.075861554
  O   -2.494309633  -5.920105995  -6.325358714
  C   -5.009554320  -5.495290560  -4.862174132
  H   -4.964809519  -5.875990092  -5.898393128
  H   -5.641872102  -6.203987676  -4.303324610
  H   -5.542303665  -4.534098611  -4.901897013
  H   -4.315130647  -4.795255981  -2.299573496
  P    2.193521200  -4.547451010   1.375133649
  O    2.153787836  -5.742072536   0.258882065
  C    3.429067546  -6.322350890  -0.146132009
  H    3.986925383  -5.553618624  -0.717757813
  H    3.996859832  -6.653929932   0.748952598
  C    3.086235267  -7.535213632  -1.022553306
  H    3.844270151  -7.676437387  -1.829247710
  O    3.227196582  -8.723131083  -0.194774696
  C    1.929166961  -9.328005173   0.029169722
  H    2.144304972 -10.410600779   0.175533284
  C    1.639261111  -7.573863642  -1.593006684
  H    0.984060471  -6.764335376  -1.189571961
  C    1.118485390  -8.971178610  -1.214675694
  H    0.019575748  -8.992961931  -1.055636684
  H    1.318201795  -9.688551831  -2.037459715
  O    1.715025989  -7.463248265  -3.016836906
  O    2.536470205  -4.791751469   2.776037858
  O    3.196548008  -3.487813808   0.595162853
  N    1.433598862  -8.738199274   1.320181401
  C    0.280139473  -7.830450577   1.418315705
  N    0.208963606  -6.972136831   2.502776703
  C    1.107394730  -7.055727170   3.523364258
  C    2.179547101  -8.025280779   3.501771282
  C    2.337194353  -8.800118631   2.390330528
  O   -0.554186030  -7.835748412   0.525815616
  N    0.924906463  -6.217033952   4.604661642
  H    1.730166999  -5.986462312   5.174773342
  H    0.264610777  -5.452630529   4.502277855
  H   -0.852889690  -6.224880586  -6.985580868
  H    3.184258355  -9.495678190   2.287250321
  H    2.850003805  -8.117265866   4.353886909
  H    1.260510981  -6.635418915  -3.293367870
  H   -3.451948481  -2.981436152  -5.039919333
  H   -3.685647550  -2.084503535  -0.360007326
  H    3.835355216  -2.961776502   1.137323958
  H   -0.528962358  -8.116276019 -10.895005597
  H   -3.612388086   5.991151108  -4.884802304
