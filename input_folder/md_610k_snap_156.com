%mem=64gb
%nproc=28       
%Chk=snap_156.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_156 

2     1 
  O    2.665010698  -4.190623036  -4.697060958
  C    3.051731158  -2.919259767  -5.227621090
  H    4.067976529  -3.105701700  -5.641379706
  H    3.114168737  -2.193591308  -4.392243403
  C    2.103070719  -2.424365121  -6.321977423
  H    2.224452078  -1.326870576  -6.499428766
  O    2.525963923  -3.013765913  -7.593553553
  C    1.491196751  -3.780329719  -8.182773055
  H    1.527816478  -3.554975103  -9.279291747
  C    0.610404579  -2.797905634  -6.131009006
  H    0.443698438  -3.461102190  -5.240184307
  C    0.183812254  -3.463583353  -7.450421178
  H   -0.439349573  -2.751283965  -8.038702931
  H   -0.464903173  -4.343595237  -7.287313712
  O   -0.105271433  -1.553381260  -6.045741345
  N    1.859043136  -5.233478017  -8.008823836
  C    1.822086298  -6.214303314  -9.016407577
  C    2.170775738  -7.449913348  -8.395682223
  N    2.449077786  -7.188019343  -7.039219379
  C    2.262052099  -5.869611229  -6.816312069
  N    1.517700632  -6.047541436 -10.341331944
  C    1.591397468  -7.191179557 -11.095567572
  N    1.901375912  -8.463550398 -10.552208246
  C    2.188826972  -8.677801799  -9.143601518
  N    1.315348493  -7.089413976 -12.434961599
  H    1.131988917  -6.171671635 -12.837491246
  H    1.428624234  -7.858334663 -13.079838374
  O    2.380938312  -9.818521528  -8.783032133
  H    2.427162424  -5.289997554  -5.826920750
  H    1.915791573  -9.303826606 -11.149759966
  P   -0.906464586  -1.301551034  -4.611281204
  O   -1.256505019   0.288851911  -4.964844242
  C   -0.399607718   1.312340663  -4.418046867
  H    0.039462228   1.820089996  -5.308535497
  H    0.414665905   0.888272299  -3.794123425
  C   -1.267083637   2.286551336  -3.615967910
  H   -0.707163660   2.704911148  -2.741617026
  O   -1.531930150   3.436571559  -4.480868452
  C   -2.889938712   3.875403330  -4.355727154
  H   -2.836271719   4.968891927  -4.117861993
  C   -2.666162689   1.772310092  -3.190210017
  H   -3.035214134   0.947082429  -3.859360781
  C   -3.571437717   3.004230226  -3.295392507
  H   -3.624624372   3.523251647  -2.307668752
  H   -4.625032084   2.752683499  -3.520909424
  O   -2.585586315   1.368741740  -1.809412405
  O   -0.037775050  -1.608352890  -3.459880670
  O   -2.302247057  -1.981244898  -5.055212217
  N   -3.513357467   3.739701728  -5.710203222
  C   -3.622777436   2.631398798  -6.548421937
  C   -4.230041718   3.090770164  -7.765009288
  N   -4.477872478   4.468926088  -7.677404314
  C   -4.048110456   4.848375027  -6.477354127
  N   -3.295764218   1.288626387  -6.392159867
  C   -3.527902027   0.447702001  -7.490525230
  N   -4.085074598   0.828565815  -8.632912437
  C   -4.466904207   2.170383364  -8.834532897
  H   -3.229923576  -0.620079380  -7.365845560
  N   -5.022999087   2.506510488 -10.012113369
  H   -5.169572163   1.817775984 -10.747578901
  H   -5.317141435   3.465566701 -10.197646541
  H   -4.081121206   5.857521800  -6.072077967
  P   -2.626730222  -0.269374772  -1.639983967
  O   -1.593710897  -0.593504005  -0.438553940
  C   -0.797430185  -1.794997769  -0.575687081
  H   -0.321421845  -1.855411429  -1.589267691
  H    0.011016311  -1.582979362   0.164558917
  C   -1.544800397  -3.079306982  -0.194437826
  H   -1.178209884  -3.459611262   0.793312871
  O   -2.971995690  -2.840097719  -0.003397470
  C   -3.733005504  -4.036095051  -0.366514415
  H   -4.190569783  -4.387811841   0.590823098
  C   -1.482176790  -4.206891376  -1.261517303
  H   -1.412713287  -3.825885923  -2.311437514
  C   -2.760672392  -5.023605504  -1.007744234
  H   -3.144648597  -5.488890547  -1.939211745
  H   -2.537850859  -5.878293026  -0.329432763
  O   -0.368060424  -5.038733369  -0.879546068
  O   -2.436881543  -0.937855801  -2.974926736
  O   -4.092336333  -0.480593475  -0.979050887
  N   -4.810278701  -3.537528597  -1.267568011
  C   -4.552169584  -3.260924961  -2.641124451
  N   -5.544386868  -2.563152283  -3.346501376
  C   -6.807207936  -2.114500617  -2.794034074
  C   -7.025889753  -2.532889712  -1.394567005
  C   -6.067828261  -3.184914168  -0.696323355
  O   -3.512262052  -3.603236158  -3.198891217
  H   -5.360423106  -2.372345724  -4.342398152
  O   -7.511650800  -1.463302187  -3.523801114
  C   -8.349037199  -2.190499194  -0.808040153
  H   -9.050883019  -1.795743768  -1.566660126
  H   -8.844824755  -3.062128657  -0.353019200
  H   -8.264546260  -1.412695050  -0.031332154
  H   -6.213744313  -3.481544868   0.351902272
  P    0.711024732  -5.404745709  -2.050198609
  O    2.052809911  -4.816075761  -1.319190726
  C    3.327588532  -5.309312625  -1.798379860
  H    3.297917256  -5.690433105  -2.845445263
  H    3.952936049  -4.386786585  -1.792037854
  C    3.848629469  -6.328948682  -0.785362618
  H    3.616884078  -6.000582670   0.258141136
  O    3.114888770  -7.590136701  -0.890165593
  C    3.965047912  -8.675367567  -1.329228495
  H    3.871687216  -9.428064855  -0.505069787
  C    5.349476266  -6.669070714  -0.996265802
  H    5.872769106  -5.963335958  -1.685402724
  C    5.377901625  -8.122052145  -1.505766170
  H    5.716067559  -8.153038266  -2.564074138
  H    6.134817704  -8.719704019  -0.957511340
  O    6.074540710  -6.496302120   0.204540330
  O    0.453059498  -4.832283340  -3.402023527
  O    0.747608622  -7.018184176  -2.004866096
  N    3.354576209  -9.234200360  -2.573122529
  C    3.338070354  -8.493733707  -3.838525761
  N    2.857068961  -9.135963300  -4.967433038
  C    2.415662468 -10.431337138  -4.919310817
  C    2.381755622 -11.151125788  -3.664946857
  C    2.848611884 -10.541010065  -2.538104191
  O    3.735308423  -7.343007582  -3.859881034
  N    1.998691321 -10.989336848  -6.089371118
  H    1.698264190 -11.951681681  -6.144077303
  H    2.110731807 -10.488997399  -6.976534267
  H    1.786097816  -4.106784200  -4.162076883
  H    2.849148246 -11.056101061  -1.564670849
  H    1.989436976 -12.167834414  -3.628132697
  H    5.790571179  -7.113853625   0.914829779
  H   -2.884624888  -2.381129354  -4.281240951
  H   -4.181546851  -1.360361937  -0.468120094
  H    1.650934439  -7.479419933  -1.803457367
  H    2.736914040  -7.923289967  -6.335333108
  H   -2.751505885   0.938606861  -5.560394093
