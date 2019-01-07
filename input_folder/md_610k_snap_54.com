%mem=64gb
%nproc=28       
%Chk=snap_54.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_54 

2     1 
  O    1.760161891  -5.840383852  -6.350845601
  C    2.705150015  -4.925214691  -6.930599724
  H    3.253190341  -5.480438689  -7.719821717
  H    3.407720915  -4.578630088  -6.148291119
  C    1.919558107  -3.756816864  -7.537644057
  H    2.544212774  -2.834802034  -7.622214383
  O    1.611711485  -4.088616902  -8.933518368
  C    0.216166674  -4.171253480  -9.155424619
  H    0.037180738  -3.710048666 -10.159649301
  C    0.565773748  -3.477352069  -6.842513427
  H    0.379509272  -4.227409954  -6.029125991
  C   -0.495244689  -3.537720378  -7.958610682
  H   -0.850348298  -2.514468609  -8.205183882
  H   -1.397362179  -4.092073909  -7.631607684
  O    0.666565215  -2.140328829  -6.325080440
  N   -0.137220014  -5.638964294  -9.281071123
  C   -0.712159204  -6.263969486 -10.410523049
  C   -0.857845556  -7.641994683 -10.088121542
  N   -0.362299508  -7.831710769  -8.781955378
  C    0.061767174  -6.630615744  -8.309129977
  N   -1.080324890  -5.683779539 -11.593404515
  C   -1.621486561  -6.550850388 -12.513481758
  N   -1.797221069  -7.931653472 -12.273778732
  C   -1.427015812  -8.584596670 -11.020074584
  N   -2.038605863  -6.022769922 -13.709899566
  H   -1.864842466  -5.040118358 -13.909024959
  H   -2.350318055  -6.595663717 -14.480930113
  O   -1.632308738  -9.766049470 -10.917463404
  H    0.519285010  -6.422531938  -7.288346690
  H   -2.213342501  -8.543749753 -12.991871476
  P   -0.201502706  -1.777352134  -4.953352531
  O   -1.480692621  -0.968553887  -5.670516565
  C   -1.300989129   0.451239353  -5.766459494
  H   -2.114949777   0.749946200  -6.459205332
  H   -0.327177758   0.702555580  -6.232520395
  C   -1.448962088   1.114993005  -4.384817774
  H   -0.547938864   0.960663591  -3.724793572
  O   -1.451862306   2.564724092  -4.599545905
  C   -2.696164386   3.151390974  -4.218623780
  H   -2.426830672   4.029948009  -3.575499830
  C   -2.778411066   0.760783292  -3.677897776
  H   -3.341152381  -0.045341084  -4.205639367
  C   -3.564891774   2.080871608  -3.566173689
  H   -3.724911231   2.303601784  -2.478572785
  H   -4.580255003   1.983717244  -3.988554840
  O   -2.516736461   0.412683038  -2.305740523
  O    0.578973858  -0.965680642  -4.010191998
  O   -0.744486448  -3.284944463  -4.670867625
  N   -3.293932813   3.681922895  -5.497215893
  C   -2.594147658   3.976331563  -6.660474271
  C   -3.538367889   4.526139254  -7.588451089
  N   -4.804417527   4.575753065  -6.989224934
  C   -4.655882819   4.084281974  -5.757857473
  N   -1.247824581   3.825941330  -6.992289737
  C   -0.878128108   4.255880343  -8.277313003
  N   -1.715161681   4.761414885  -9.171302177
  C   -3.085519700   4.938748661  -8.882010628
  H    0.194193486   4.154252746  -8.558586425
  N   -3.872016405   5.475629502  -9.828552858
  H   -3.497099642   5.768177466 -10.730897970
  H   -4.866404202   5.633245070  -9.659798193
  H   -5.433719170   3.996488189  -5.003258223
  P   -2.268104211  -1.189696562  -1.996308164
  O   -1.349223330  -0.979765849  -0.677454305
  C   -0.505950692  -2.107302893  -0.308112454
  H    0.124900228  -2.434487640  -1.156841177
  H    0.147758908  -1.666114705   0.483709409
  C   -1.465025106  -3.158601191   0.236443992
  H   -2.041608445  -2.781970486   1.117931350
  O   -2.456844449  -3.291453603  -0.845618168
  C   -2.785593971  -4.697715598  -1.075088052
  H   -3.902138912  -4.715198512  -1.007493599
  C   -0.923810306  -4.582031436   0.493497354
  H    0.043197970  -4.778649823  -0.034134447
  C   -2.070003274  -5.493517093   0.011613598
  H   -1.699930891  -6.484384773  -0.315636750
  H   -2.744188595  -5.717180785   0.872935123
  O   -0.834024923  -4.842874892   1.894151861
  O   -1.728544553  -1.894103846  -3.199207204
  O   -3.804581623  -1.592461260  -1.627167200
  N   -2.379794079  -5.053161869  -2.462962256
  C   -1.107324376  -5.647225109  -2.766195809
  N   -0.937888562  -6.098457205  -4.091464629
  C   -1.853691670  -5.870635536  -5.165143697
  C   -3.084235095  -5.154600563  -4.788776895
  C   -3.305055670  -4.776092184  -3.506012876
  O   -0.224317504  -5.777078100  -1.935672145
  H   -0.027481375  -6.559773634  -4.310935967
  O   -1.538869509  -6.283458977  -6.268097216
  C   -4.033627412  -4.834206172  -5.889559971
  H   -3.830527980  -5.439170813  -6.789955632
  H   -5.081652687  -5.025575400  -5.612707864
  H   -3.950700595  -3.775076459  -6.183789729
  H   -4.203820759  -4.215873797  -3.209350994
  P    0.528073589  -4.395217063   2.727458110
  O    1.572345448  -5.657040890   2.627242592
  C    1.859509343  -6.364363588   1.415115680
  H    2.088751903  -7.390345643   1.792962287
  H    0.998758078  -6.431290667   0.729843807
  C    3.118107222  -5.739671819   0.796753450
  H    3.808514034  -5.396467798   1.606561245
  O    3.858273755  -6.804121046   0.135241392
  C    3.958059127  -6.569977560  -1.277363257
  H    4.979561952  -6.924245976  -1.557327647
  C    2.866672088  -4.646069821  -0.273323603
  H    1.782808259  -4.494787725  -0.511054162
  C    3.677710401  -5.083463869  -1.496316042
  H    3.144801038  -4.860366929  -2.450834421
  H    4.622502236  -4.511847377  -1.594747932
  O    3.246494031  -3.365893131   0.228412153
  O    0.301165759  -3.966288114   4.091994125
  O    1.061975732  -3.320598894   1.600293877
  N    2.942067334  -7.489985606  -1.922088735
  C    2.398204188  -7.230042735  -3.244619204
  N    1.443161890  -8.088928430  -3.762826615
  C    1.055940219  -9.192634947  -3.054155282
  C    1.622301554  -9.493728382  -1.765600518
  C    2.547555280  -8.635245999  -1.234548954
  O    2.799600432  -6.254141860  -3.874361620
  N    0.078516966  -9.978447811  -3.611454795
  H   -0.172646914 -10.868089436  -3.207278564
  H   -0.255396736  -9.794198078  -4.546731729
  H    2.104732966  -6.248789124  -5.491682094
  H    3.016013964  -8.812814618  -0.249410740
  H    1.325619130 -10.391348335  -1.225759525
  H    4.216304530  -3.291648664   0.388552969
  H   -1.291823026  -3.389521825  -3.798327724
  H   -3.987157603  -2.000176793  -0.728802703
  H    2.043406366  -3.093421456   1.507510691
  H   -0.367947289  -8.722324887  -8.299711772
  H   -0.592384230   3.453186378  -6.282749708

