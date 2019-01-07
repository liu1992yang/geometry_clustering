%mem=64gb
%nproc=28       
%Chk=snap_110.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_110 

2     1 
  O    3.420683391  -4.264927213  -6.311193558
  C    3.447712383  -3.067737715  -7.082521259
  H    4.097233817  -3.223558516  -7.967719516
  H    3.952268766  -2.348759610  -6.399912566
  C    2.067558735  -2.561966120  -7.506954114
  H    2.072200944  -1.461790440  -7.713026546
  O    1.769521864  -3.145829773  -8.821555551
  C    0.640940852  -3.999348450  -8.760582878
  H    0.065640004  -3.833106978  -9.705049703
  C    0.896682031  -2.946569580  -6.574438133
  H    1.236376457  -3.520191861  -5.667406760
  C   -0.087729444  -3.743331553  -7.444647455
  H   -1.032692322  -3.178254380  -7.610900852
  H   -0.412723534  -4.678352103  -6.935425279
  O    0.318989243  -1.680356075  -6.191659864
  N    1.193409335  -5.413997048  -8.820602565
  C    0.494039522  -6.601419905  -8.523509219
  C    1.423778253  -7.672581671  -8.677864644
  N    2.664607062  -7.119919726  -9.016189733
  C    2.518473539  -5.768176654  -9.111920004
  N   -0.835304250  -6.737116789  -8.229344147
  C   -1.245655441  -8.022814142  -8.010425593
  N   -0.410359885  -9.145384846  -8.181056447
  C    1.020647032  -9.046558528  -8.446062333
  N   -2.547569469  -8.223172261  -7.555390191
  H   -3.105601567  -7.353314585  -7.432850566
  H   -3.086663526  -8.998974926  -7.928521055
  O    1.671263770 -10.057755160  -8.469533976
  H    3.286608731  -5.040766783  -9.391564368
  H   -0.758580227 -10.093051676  -7.977526522
  P   -0.547651753  -1.710568852  -4.778134422
  O   -0.768827443  -0.066645665  -4.716566178
  C    0.023495768   0.641053810  -3.730843157
  H    0.654770917   1.343964713  -4.323786219
  H    0.681584095  -0.039758164  -3.148010687
  C   -0.932371286   1.409823139  -2.819339795
  H   -0.541394860   1.457992746  -1.770380923
  O   -0.920296255   2.794204491  -3.291168008
  C   -2.229236356   3.370974004  -3.224422840
  H   -2.133272181   4.280144565  -2.577327782
  C   -2.423465241   0.976298331  -2.835980211
  H   -2.708270578   0.405908405  -3.753391917
  C   -3.195462738   2.297999269  -2.715126504
  H   -3.465388317   2.467038802  -1.642930717
  H   -4.169605050   2.277296690  -3.235974342
  O   -2.711263787   0.228330685  -1.652608309
  O    0.225933606  -2.325223340  -3.678794209
  O   -1.925609144  -2.246787855  -5.450656935
  N   -2.562147744   3.837454734  -4.605562638
  C   -2.539400927   3.146432902  -5.813685871
  C   -2.946995751   4.076651030  -6.827557818
  N   -3.200076427   5.327718407  -6.246507953
  C   -2.966027077   5.187962703  -4.944517650
  N   -2.231344195   1.829941722  -6.145355249
  C   -2.307397537   1.491503879  -7.505925870
  N   -2.688473232   2.313716990  -8.472950410
  C   -3.025647877   3.654865709  -8.192520482
  H   -2.036847476   0.444288587  -7.771266806
  N   -3.399043431   4.449056731  -9.209649189
  H   -3.437980133   4.106004385 -10.168664338
  H   -3.646446306   5.426809335  -9.047756006
  H   -3.056859837   5.964237993  -4.186897389
  P   -2.502278469  -1.416341896  -1.789367506
  O   -1.307670266  -1.502367299  -0.694605933
  C   -0.501438378  -2.715562959  -0.724578375
  H   -0.114527354  -2.898649714  -1.755531452
  H    0.346511484  -2.470169577  -0.044554642
  C   -1.396085996  -3.824897846  -0.191835316
  H   -1.670663942  -3.686617601   0.881415963
  O   -2.650916497  -3.600522480  -0.926089786
  C   -3.141930851  -4.838353455  -1.522105962
  H   -4.209257182  -4.869705671  -1.186231435
  C   -0.969664790  -5.278463315  -0.492427500
  H   -0.132970611  -5.353141389  -1.244288956
  C   -2.263292942  -5.960265120  -0.971849420
  H   -2.048592993  -6.774111448  -1.691782339
  H   -2.747282196  -6.471345484  -0.104638623
  O   -0.650000730  -5.948276834   0.740252559
  O   -2.208706490  -1.773816910  -3.220311529
  O   -3.995506885  -1.825847535  -1.296233005
  N   -3.104997602  -4.703385241  -3.003048554
  C   -2.005343482  -5.204357640  -3.778853461
  N   -2.293894347  -5.477021843  -5.135350850
  C   -3.526188136  -5.137099531  -5.788802255
  C   -4.420096636  -4.276996667  -4.990940953
  C   -4.228200419  -4.134144407  -3.656708291
  O   -0.909565289  -5.425723817  -3.301738817
  H   -1.598837061  -6.052713420  -5.654374681
  O   -3.715351875  -5.585950400  -6.900185287
  C   -5.548539198  -3.641628644  -5.722481769
  H   -5.900918351  -2.714271476  -5.252232860
  H   -5.272896847  -3.397440646  -6.762999218
  H   -6.413934583  -4.325588959  -5.790636225
  H   -4.919583731  -3.565371661  -3.015415447
  P    0.904702350  -5.872182351   1.264691321
  O    1.651550112  -6.139137978  -0.178512219
  C    3.075751419  -5.850381890  -0.229952807
  H    3.402120394  -5.197414532   0.604285908
  H    3.586345861  -6.838382699  -0.163004989
  C    3.355321739  -5.178743382  -1.572631318
  H    4.310396805  -4.597931122  -1.529215462
  O    3.610052603  -6.250429540  -2.546578184
  C    2.948812071  -5.978192565  -3.785810436
  H    3.700670260  -6.188730260  -4.582032087
  C    2.217830097  -4.332504597  -2.191096937
  H    1.205824616  -4.654975205  -1.837789936
  C    2.397876600  -4.555125848  -3.699234653
  H    1.455949352  -4.400483430  -4.268331952
  H    3.128515691  -3.833502176  -4.131744675
  O    2.432490313  -2.975253462  -1.814531164
  O    1.248856971  -6.660469301   2.435161801
  O    1.113033177  -4.233911731   1.394149043
  N    1.807561044  -6.966162258  -3.936777197
  C    1.052860535  -6.946653772  -5.181201319
  N   -0.032127019  -7.788873308  -5.317122710
  C   -0.424136383  -8.569357127  -4.258230857
  C    0.272664465  -8.557465875  -3.008649512
  C    1.370077009  -7.738379725  -2.872837565
  O    1.408293330  -6.175266217  -6.072340702
  N   -1.572552176  -9.304242106  -4.456432404
  H   -1.830972405 -10.043940097  -3.820772026
  H   -1.977409274  -9.332866432  -5.382829729
  H    2.793067735  -4.940354336  -6.666956807
  H    1.938680959  -7.662832289  -1.925717509
  H   -0.054441449  -9.179920772  -2.178262810
  H    1.857353859  -2.394784901  -2.384810603
  H   -2.667149414  -2.517666734  -4.789704651
  H   -4.104417820  -2.145667374  -0.346791208
  H    1.393764240  -3.867245408   2.273190200
  H    3.509281583  -7.664128677  -9.174774862
  H   -1.842640120   1.162498550  -5.442950834
