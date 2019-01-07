%mem=64gb
%nproc=28       
%Chk=snap_165.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_165 

2     1 
  O    3.229915920  -4.341854480  -4.369967871
  C    3.569176725  -3.069681135  -4.933028066
  H    4.583654423  -3.246295741  -5.355138367
  H    3.639955166  -2.331114199  -4.109301219
  C    2.601246002  -2.592554422  -6.019106276
  H    2.790531088  -1.521782980  -6.291117328
  O    2.895532004  -3.301780505  -7.260457123
  C    1.801824250  -4.104203456  -7.675553451
  H    1.674230479  -3.899461555  -8.771211610
  C    1.107311927  -2.834965393  -5.703707615
  H    0.949008662  -3.222955048  -4.665269170
  C    0.598562966  -3.804896028  -6.778597931
  H   -0.234595736  -3.352071569  -7.360247433
  H    0.155621036  -4.711033975  -6.306623223
  O    0.484877192  -1.539187768  -5.882679335
  N    2.236233326  -5.543853679  -7.547322578
  C    2.082074469  -6.518346648  -8.550074755
  C    2.584515662  -7.741674309  -8.016677892
  N    3.047526097  -7.481805308  -6.710924227
  C    2.832880959  -6.173569366  -6.437396859
  N    1.549138485  -6.348382079  -9.799597186
  C    1.523022629  -7.480989790 -10.572220391
  N    1.997732518  -8.737754862 -10.120651155
  C    2.566806793  -8.948400019  -8.799869008
  N    0.983616645  -7.383301481 -11.828552198
  H    0.686779066  -6.472452147 -12.175008946
  H    0.985973547  -8.146341877 -12.489621591
  O    2.934341748 -10.072604262  -8.536967024
  H    3.096462967  -5.606787838  -5.482923658
  H    1.954949498  -9.568007549 -10.730385651
  P   -0.819732580  -1.254408531  -4.921081868
  O   -1.362808914   0.141351612  -5.632643307
  C   -0.633491520   1.356270437  -5.320196446
  H   -0.657814322   1.916762828  -6.285545161
  H    0.421606804   1.143654213  -5.055957097
  C   -1.348341260   2.143786964  -4.220709824
  H   -0.618871214   2.608729211  -3.511548129
  O   -2.025588228   3.262819147  -4.881389909
  C   -3.313615487   3.498709604  -4.293081338
  H   -3.329172260   4.583383786  -4.019395322
  C   -2.472787490   1.406768797  -3.449989687
  H   -2.963889290   0.620448500  -4.087003645
  C   -3.488550405   2.514268830  -3.134887178
  H   -3.263382833   2.980223528  -2.149818216
  H   -4.528991423   2.131429856  -3.026338031
  O   -1.891195451   0.874959581  -2.263616000
  O   -0.383239301  -1.145837705  -3.491172262
  O   -1.865363051  -2.305391917  -5.524265824
  N   -4.324479699   3.294996045  -5.380652088
  C   -4.535526942   2.207320180  -6.221561419
  C   -5.638214822   2.550596109  -7.076044379
  N   -6.101123065   3.834863982  -6.753074610
  C   -5.329041159   4.268533327  -5.759572365
  N   -3.907704830   0.970586846  -6.352551139
  C   -4.384113022   0.119624183  -7.359108881
  N   -5.408381547   0.388096572  -8.154797601
  C   -6.085647238   1.623577121  -8.066031697
  H   -3.856178761  -0.855021941  -7.474926435
  N   -7.095527953   1.855475802  -8.923517213
  H   -7.373047752   1.162997336  -9.616128154
  H   -7.615087952   2.732885754  -8.895269100
  H   -5.407100310   5.230166198  -5.256705997
  P   -2.166978094  -0.759363154  -2.081252974
  O   -0.984232938  -1.192473451  -1.060258423
  C   -0.441130934  -2.523560025  -1.213766690
  H    0.181171008  -2.574252286  -2.131756581
  H    0.234935480  -2.582917412  -0.330908117
  C   -1.569507677  -3.557348221  -1.149448444
  H   -1.519551034  -4.158482133  -0.209411081
  O   -2.807291544  -2.782613210  -1.104073427
  C   -3.925058237  -3.650504823  -1.559002743
  H   -4.407585967  -4.001352672  -0.611419226
  C   -1.775735206  -4.483929264  -2.375325594
  H   -1.430101416  -4.054198008  -3.342125692
  C   -3.282959860  -4.779000171  -2.368816607
  H   -3.690127302  -4.877246166  -3.402597725
  H   -3.472979130  -5.775800360  -1.906713701
  O   -1.134423035  -5.749895368  -2.088612989
  O   -2.641262720  -1.332962527  -3.413267597
  O   -3.428122933  -0.559804427  -1.036297791
  N   -4.871260860  -2.756085240  -2.267210446
  C   -4.689904514  -2.434603432  -3.647180447
  N   -5.294856585  -1.247605594  -4.116590789
  C   -5.920054145  -0.261702064  -3.275045316
  C   -6.213361012  -0.738953733  -1.917283708
  C   -5.655303761  -1.882452020  -1.450333282
  O   -4.142742158  -3.195786333  -4.444422046
  H   -5.138622482  -1.007358791  -5.097125971
  O   -6.105323181   0.830925258  -3.773531793
  C   -7.102610514   0.119974891  -1.089037114
  H   -6.523470145   0.716789068  -0.365298318
  H   -7.674983225   0.836676872  -1.704130924
  H   -7.845363603  -0.461141849  -0.521785717
  H   -5.796206145  -2.215138329  -0.413306063
  P    0.375471591  -5.916696871  -2.686949094
  O    1.290767369  -5.336628224  -1.448225615
  C    2.695170567  -5.692698443  -1.478690256
  H    3.063061400  -5.961918372  -2.496262127
  H    3.191012119  -4.744465993  -1.167872631
  C    2.941606955  -6.789768805  -0.442054206
  H    2.428477856  -6.567490099   0.530108101
  O    2.315498372  -8.040377589  -0.869234755
  C    3.269409108  -9.126809976  -0.915174961
  H    2.869961654  -9.863514722  -0.172323950
  C    4.451470379  -7.078789615  -0.242072378
  H    5.117801474  -6.430105203  -0.850357576
  C    4.650120052  -8.570837802  -0.564771091
  H    5.382147188  -8.711520884  -1.384665274
  H    5.082185696  -9.101215109   0.310647151
  O    4.657261323  -6.818586105   1.145526013
  O    0.655823240  -5.119247790  -3.919151620
  O    0.525605083  -7.511214591  -2.774693107
  N    3.186655935  -9.712968897  -2.288171195
  C    3.450216741  -8.916845860  -3.490183736
  N    3.420503066  -9.551897401  -4.717195526
  C    3.212408798 -10.904968782  -4.811752781
  C    2.940113635 -11.696216534  -3.631208040
  C    2.928503114 -11.084705112  -2.412659223
  O    3.653341763  -7.719340004  -3.371886479
  N    3.275983954 -11.455633021  -6.052358644
  H    3.124572441 -12.440536809  -6.213277958
  H    3.436276994 -10.873678309  -6.880648006
  H    2.279242673  -4.353943059  -4.019392418
  H    2.720679216 -11.647846192  -1.489158747
  H    2.745813594 -12.765915923  -3.721693412
  H    5.611827575  -6.741991856   1.366538484
  H   -2.718508413  -2.536603084  -4.945465328
  H   -3.243526094  -0.749308191  -0.074309117
  H    1.356497059  -7.932860531  -2.294830154
  H    3.413947524  -8.217376125  -6.053900347
  H   -3.075581373   0.707882072  -5.770108197
