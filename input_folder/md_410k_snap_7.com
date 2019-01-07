%mem=64gb
%nproc=28       
%Chk=snap_7.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_7 

2     1 
  O    1.917344475  -3.695592538  -7.636826591
  C    2.087585448  -2.622187901  -8.574310556
  H    2.834933473  -3.027713368  -9.293323678
  H    2.515275714  -1.745980646  -8.048257430
  C    0.784187737  -2.262097514  -9.295997598
  H    0.786713807  -1.200572515  -9.641072932
  O    0.748283534  -3.037412367 -10.542702525
  C   -0.450869186  -3.787997576 -10.642697404
  H   -0.731488346  -3.785248625 -11.725406818
  C   -0.530435423  -2.618673569  -8.555395818
  H   -0.364902970  -3.394974641  -7.753448938
  C   -1.447950113  -3.186043637  -9.645760021
  H   -2.055640643  -2.377973356 -10.109845237
  H   -2.185437068  -3.909429367  -9.249031147
  O   -1.119803818  -1.399274374  -8.035071485
  N   -0.120519821  -5.212305416 -10.254568605
  C   -0.490800355  -6.365674688 -10.980803983
  C   -0.031958001  -7.485307081 -10.231940251
  N    0.613992071  -6.995563090  -9.078252349
  C    0.559723641  -5.640323983  -9.101054976
  N   -1.158472333  -6.417716115 -12.171404010
  C   -1.363806939  -7.689956451 -12.657717697
  N   -0.949098839  -8.855009898 -11.975761857
  C   -0.244592083  -8.836336627 -10.692814575
  N   -2.023636355  -7.805118571 -13.853237095
  H   -2.298256994  -6.961663767 -14.355342050
  H   -2.166014838  -8.689078646 -14.321486205
  O    0.053229140  -9.892745601 -10.201087217
  H    1.006370465  -4.927660693  -8.314792986
  H   -1.131755355  -9.790323151 -12.370672966
  P   -0.905550821  -1.330679618  -6.403988862
  O   -1.359419384   0.223344621  -6.100186025
  C   -0.754312852   0.811101261  -4.917816030
  H   -0.384294373   1.796725946  -5.286504433
  H    0.112345074   0.220996233  -4.544691607
  C   -1.809080273   1.019805938  -3.825097756
  H   -1.348408240   0.950557952  -2.801802402
  O   -2.214763816   2.424646911  -3.865004307
  C   -3.569330891   2.561873599  -4.270099934
  H   -3.995862274   3.359819231  -3.618543316
  C   -3.106670189   0.176740417  -3.949202971
  H   -3.088034674  -0.583166771  -4.768814316
  C   -4.243892816   1.196648554  -4.144599233
  H   -4.913410737   1.160112557  -3.247323483
  H   -4.904274925   0.933677157  -4.991281439
  O   -3.420844160  -0.471573498  -2.706249960
  O    0.449308052  -1.743552006  -5.956182142
  O   -2.174073893  -2.169199499  -5.863601422
  N   -3.540570024   3.062421989  -5.688245296
  C   -3.592079869   4.391516721  -6.106787769
  C   -3.330514204   4.393015748  -7.516061577
  N   -3.088741885   3.080218361  -7.950453677
  C   -3.195563289   2.305946115  -6.872344901
  N   -3.861082762   5.582583271  -5.430575135
  C   -3.847160367   6.759620519  -6.208186833
  N   -3.608895613   6.800612097  -7.507010953
  C   -3.332971006   5.626191681  -8.240596801
  H   -4.055167229   7.716077030  -5.676059131
  N   -3.087127358   5.740287564  -9.555764319
  H   -3.101830881   6.646651751 -10.024550276
  H   -2.885852416   4.917633876 -10.124750670
  H   -3.035296200   1.224404105  -6.831456139
  P   -2.450825905  -1.731641023  -2.234376999
  O   -1.506089671  -0.817597950  -1.253516288
  C   -0.250162593  -1.393918427  -0.825397940
  H    0.399038897  -1.597168468  -1.710226701
  H    0.196219934  -0.581455394  -0.214989240
  C   -0.569041126  -2.641227142  -0.003985460
  H   -1.101250855  -2.410712808   0.953266086
  O   -1.558588664  -3.365807139  -0.810455268
  C   -1.093619329  -4.724313131  -1.129592152
  H   -1.970552469  -5.359753564  -0.846175656
  C    0.615832046  -3.598725431   0.240187508
  H    1.577635397  -3.247102561  -0.209961799
  C    0.163759452  -4.961306469  -0.298537349
  H    0.981879412  -5.453907195  -0.886359974
  H   -0.040093509  -5.676828845   0.533917699
  O    0.705721895  -3.608299025   1.686411864
  O   -1.739818400  -2.361916026  -3.385742410
  O   -3.623033885  -2.530780049  -1.460212034
  N   -0.869615591  -4.805634463  -2.594489874
  C    0.182563435  -4.054446799  -3.213369943
  N    0.274584562  -4.172246073  -4.620875217
  C   -0.527913599  -5.032948210  -5.421181334
  C   -1.511562606  -5.828504718  -4.700117200
  C   -1.652887255  -5.703280371  -3.351369076
  O    0.972635691  -3.372627387  -2.585333201
  H    0.934282587  -3.516391233  -5.093554208
  O   -0.316991081  -5.002775922  -6.634207314
  C   -2.323192447  -6.802016379  -5.485138969
  H   -2.097649609  -6.747204931  -6.560917710
  H   -2.121987262  -7.837454619  -5.164130998
  H   -3.402127277  -6.622814800  -5.374419085
  H   -2.369177454  -6.321944603  -2.785121752
  P    2.033959099  -4.297661208   2.355382628
  O    1.884314790  -5.800065841   1.695334524
  C    2.728102014  -6.840788908   2.246667798
  H    3.250523681  -6.522537229   3.171399313
  H    2.003148095  -7.641530517   2.515388025
  C    3.708861378  -7.344225266   1.179506701
  H    4.660947352  -7.687381559   1.652478358
  O    3.180823145  -8.576946875   0.606143118
  C    2.603356727  -8.343657211  -0.688794225
  H    2.950085326  -9.200941869  -1.314916705
  C    3.962345027  -6.403272840  -0.026719950
  H    3.780294177  -5.321306648   0.184427751
  C    3.070121729  -6.959934473  -1.150380604
  H    2.222008779  -6.263319287  -1.355430502
  H    3.609837706  -7.007500121  -2.116456037
  O    5.336944591  -6.398350120  -0.369724173
  O    2.221674872  -4.153512633   3.792101760
  O    3.224968330  -3.695168927   1.380068940
  N    1.106353456  -8.464340776  -0.513731550
  C    0.180372140  -7.914972769  -1.515899359
  N   -1.179432307  -7.985716236  -1.250467384
  C   -1.633889507  -8.648990280  -0.146353095
  C   -0.738374110  -9.246064036   0.804478481
  C    0.612956644  -9.137680574   0.593442182
  O    0.662011904  -7.390027686  -2.503422122
  N   -2.998520298  -8.667333485   0.045566010
  H   -3.408419071  -9.256795658   0.754883163
  H   -3.616384949  -8.370451048  -0.697400669
  H    1.464963566  -3.359199789  -6.807009646
  H    1.350148933  -9.581756371   1.287092637
  H   -1.119892942  -9.777248546   1.673549102
  H    5.663659669  -7.285340548  -0.638171558
  H   -2.052036248  -2.657262512  -4.927461023
  H   -3.417364774  -2.916134171  -0.546797771
  H    3.983181233  -3.207800312   1.796115527
  H    1.022615103  -7.586930049  -8.360245071
  H   -4.024158711   5.602752490  -4.420569464

