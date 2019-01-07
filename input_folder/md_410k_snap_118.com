%mem=64gb
%nproc=28       
%Chk=snap_118.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_118 

2     1 
  O    2.788701376  -2.686821590  -6.442323659
  C    2.508219087  -3.736382720  -7.381208132
  H    2.504078140  -4.699515412  -6.825769646
  H    3.373175899  -3.721920898  -8.076034322
  C    1.184300790  -3.453959488  -8.092281171
  H    1.202908362  -2.475425590  -8.630253031
  O    0.981179371  -4.461619851  -9.134014124
  C   -0.357607799  -4.957885869  -9.101196517
  H   -0.721725896  -4.983610515 -10.156379574
  C   -0.054172536  -3.560128946  -7.162437731
  H    0.130352025  -4.222150004  -6.283818306
  C   -1.145530487  -4.122495680  -8.088377284
  H   -1.709425270  -3.301575767  -8.579137430
  H   -1.921410704  -4.698909900  -7.533697965
  O   -0.317346779  -2.213659414  -6.746575623
  N   -0.249483943  -6.386092458  -8.616459878
  C   -1.224542555  -7.394139812  -8.765635919
  C   -0.861921025  -8.449911171  -7.876105862
  N    0.303687777  -8.063478447  -7.197312898
  C    0.659010139  -6.831485964  -7.640871053
  N   -2.293603260  -7.404159470  -9.617681742
  C   -3.068977111  -8.534704450  -9.546200161
  N   -2.807945320  -9.608088411  -8.661812689
  C   -1.682828463  -9.627978290  -7.733827139
  N   -4.170267073  -8.596377328 -10.364749285
  H   -4.322834818  -7.853930111 -11.045151280
  H   -4.725221669  -9.432575517 -10.471501016
  O   -1.566909658 -10.574856163  -6.996754874
  H    1.538532826  -6.246187296  -7.314289768
  H   -3.443157688 -10.417457588  -8.617594864
  P   -1.075442749  -2.063775615  -5.267830128
  O   -1.858800090  -0.647082243  -5.602086518
  C   -1.244457366   0.564543092  -5.133247869
  H   -1.462160088   1.284531424  -5.950731880
  H   -0.146555620   0.478730356  -5.010984244
  C   -1.937441684   0.977819746  -3.822521150
  H   -1.345470262   0.703653393  -2.911658293
  O   -1.942836834   2.438876692  -3.761587956
  C   -3.273985409   2.962917148  -3.807355052
  H   -3.315438063   3.728044908  -2.988150896
  C   -3.417338565   0.517682060  -3.752892126
  H   -3.710013026  -0.147319607  -4.606085203
  C   -4.257636026   1.804993120  -3.686887807
  H   -4.794208630   1.829961152  -2.702610775
  H   -5.063852195   1.797718979  -4.442663977
  O   -3.709258532  -0.142751802  -2.510907331
  O   -0.041071396  -2.039683421  -4.208054596
  O   -2.178855201  -3.224217019  -5.527185565
  N   -3.369537071   3.686992114  -5.123274036
  C   -2.305712197   4.292755473  -5.783839629
  C   -2.850412258   4.966157513  -6.924736697
  N   -4.238141468   4.771768112  -6.960377501
  C   -4.540833314   4.023185729  -5.899047343
  N   -0.940178565   4.322061623  -5.502116862
  C   -0.142636240   5.066710036  -6.390066271
  N   -0.601200183   5.709090960  -7.452330641
  C   -1.974408169   5.705846670  -7.783529057
  H    0.949288310   5.109134581  -6.176001837
  N   -2.368608458   6.389773297  -8.868318656
  H   -1.703587174   6.909056742  -9.442990189
  H   -3.351742107   6.423021403  -9.142545623
  H   -5.536171091   3.695089225  -5.610688609
  P   -2.957109497  -1.589147398  -2.223287556
  O   -1.757346151  -0.951181631  -1.319672202
  C   -0.609023629  -1.820246063  -1.100320997
  H   -0.220511446  -2.204722676  -2.074243846
  H    0.145570624  -1.138120541  -0.655927709
  C   -1.099881487  -2.893016842  -0.137795614
  H   -1.401637735  -2.480330617   0.855361922
  O   -2.349812367  -3.347757668  -0.768753542
  C   -2.381555763  -4.809336051  -0.881096855
  H   -3.398373098  -5.065055592  -0.488893893
  C   -0.223370496  -4.146924165   0.026004454
  H    0.561577930  -4.220485739  -0.782694809
  C   -1.225322300  -5.317273764  -0.028777779
  H   -0.749211347  -6.251339008  -0.404369038
  H   -1.570041796  -5.583836288   0.996463339
  O    0.383749952  -4.019210782   1.315911693
  O   -2.524540700  -2.251195126  -3.495419827
  O   -4.211046703  -2.269705745  -1.453094078
  N   -2.320328911  -5.172064196  -2.325031195
  C   -1.090430687  -5.494805845  -2.990935244
  N   -1.197597207  -5.945886225  -4.323890808
  C   -2.418508490  -6.040606562  -5.067910042
  C   -3.615549946  -5.589952910  -4.337519319
  C   -3.541303514  -5.185922250  -3.045110311
  O   -0.004625461  -5.419782338  -2.444291574
  H   -0.312699573  -6.251086620  -4.787667781
  O   -2.344182403  -6.453611414  -6.208403252
  C   -4.887587366  -5.601432545  -5.110312175
  H   -4.770944906  -5.085106832  -6.078859003
  H   -5.197353580  -6.633873988  -5.347894983
  H   -5.723352796  -5.123532374  -4.582033760
  H   -4.428317289  -4.840784169  -2.490288179
  P    1.680736155  -5.038552512   1.552969226
  O    2.814281019  -4.248506664   0.649136546
  C    3.468142583  -5.044759317  -0.367336441
  H    4.439247011  -5.372021471   0.064830790
  H    2.875852207  -5.939538214  -0.654219378
  C    3.699472533  -4.096296174  -1.553479338
  H    4.364406065  -3.239819252  -1.296091018
  O    4.465688493  -4.904656780  -2.511451555
  C    3.744460881  -5.087684677  -3.729563364
  H    4.513266852  -5.040137508  -4.537233902
  C    2.429195037  -3.629646729  -2.299189041
  H    1.497187186  -4.088233974  -1.871562068
  C    2.654794115  -4.025185758  -3.762607746
  H    1.709318808  -4.345380100  -4.245031395
  H    3.002721846  -3.156802747  -4.373926201
  O    2.305042592  -2.214322284  -2.140970844
  O    1.456145764  -6.447270058   1.223620122
  O    2.078084939  -4.615489609   3.061669189
  N    3.185794592  -6.506915542  -3.709664112
  C    2.112966310  -6.890414568  -4.591469566
  N    1.648025301  -8.191429558  -4.593442674
  C    2.177928974  -9.109990514  -3.716986396
  C    3.242365881  -8.741635373  -2.821341012
  C    3.727226065  -7.460575566  -2.849985569
  O    1.624003181  -6.051698718  -5.361534186
  N    1.664709491 -10.373721960  -3.770636516
  H    1.974932406 -11.101087832  -3.141847005
  H    0.908872219 -10.606752176  -4.403355980
  H    1.951897759  -2.266801408  -6.133749755
  H    4.567915787  -7.132988441  -2.211655424
  H    3.661381855  -9.473869986  -2.130127500
  H    1.785150803  -1.844892452  -2.904066088
  H   -2.814064036  -3.425959956  -4.736282122
  H   -4.161738728  -2.338513872  -0.447226688
  H    2.376603826  -3.669878574   3.219633836
  H    0.737723900  -8.566102159  -6.404739619
  H   -0.581429551   3.815794730  -4.676466983

