%mem=64gb
%nproc=28       
%Chk=snap_60.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_60 

2     1 
  O    1.047378854  -6.687576761  -7.013824199
  C    2.045444209  -5.720603072  -7.354226989
  H    2.360951497  -6.011887286  -8.379948875
  H    2.900571045  -5.801521029  -6.658307618
  C    1.474834769  -4.291612834  -7.377372099
  H    2.264433199  -3.516442450  -7.240718269
  O    1.000380667  -4.047838870  -8.743878399
  C   -0.412843129  -4.078367324  -8.796572420
  H   -0.693122418  -3.392318995  -9.635653242
  C    0.259613691  -4.049870585  -6.452448227
  H    0.069494188  -4.922746958  -5.782658469
  C   -0.919287873  -3.729095920  -7.399605388
  H   -1.181699834  -2.647177979  -7.355409536
  H   -1.849149079  -4.248576145  -7.091923173
  O    0.616053075  -2.918207069  -5.620242103
  N   -0.836425845  -5.470450858  -9.213224903
  C   -0.845694760  -5.931534188 -10.551668300
  C   -1.083574915  -7.333405951 -10.505438364
  N   -1.186156317  -7.707230816  -9.150488804
  C   -1.024441492  -6.595074595  -8.387236573
  N   -0.693986947  -5.186861853 -11.685914906
  C   -0.768550293  -5.905422750 -12.858379834
  N   -1.001046381  -7.296598071 -12.902544454
  C   -1.177790928  -8.124766776 -11.708786478
  N   -0.623506694  -5.209232742 -14.030850640
  H   -0.418667421  -4.212799087 -13.996966640
  H   -0.586942554  -5.655931497 -14.936073316
  O   -1.377312263  -9.300115866 -11.866146284
  H   -1.047680536  -6.562484051  -7.275111844
  H   -1.066710537  -7.793355732 -13.804431828
  P   -0.507819505  -2.444727247  -4.500802629
  O   -1.266552367  -1.220337249  -5.339881244
  C   -0.938015984   0.124012064  -4.977876822
  H   -1.150346042   0.681755311  -5.914569386
  H    0.133886689   0.243329335  -4.722374710
  C   -1.847535377   0.598169806  -3.826714294
  H   -1.395577856   0.429880893  -2.818446912
  O   -1.905773917   2.059003236  -3.894765564
  C   -3.188978160   2.511340296  -4.332756792
  H   -3.425430963   3.393397887  -3.681782685
  C   -3.307601116   0.090919638  -3.955052832
  H   -3.415137562  -0.715646738  -4.722099991
  C   -4.159842723   1.339404928  -4.257113491
  H   -4.889072090   1.476376652  -3.414033385
  H   -4.792553596   1.195804240  -5.149307937
  O   -3.841405408  -0.365366302  -2.701059334
  O    0.156998928  -2.036448634  -3.245592208
  O   -1.549590902  -3.663254543  -4.759543578
  N   -2.971654636   3.010636557  -5.738341017
  C   -1.767549635   3.504039352  -6.230009818
  C   -2.010890203   3.939687187  -7.572372791
  N   -3.354595459   3.710859722  -7.898338632
  C   -3.917598058   3.168478588  -6.817074301
  N   -0.506023112   3.608961541  -5.644805285
  C    0.495478951   4.195071140  -6.437901770
  N    0.314961629   4.620292573  -7.678488753
  C   -0.940650603   4.531169282  -8.317770572
  H    1.506568043   4.299132816  -5.982519004
  N   -1.050242846   5.001217526  -9.569967383
  H   -0.257788128   5.430578189 -10.049527019
  H   -1.940531889   4.969577465 -10.068223735
  H   -4.962127630   2.883322994  -6.715395883
  P   -3.158984021  -1.716552780  -2.032397933
  O   -2.359152806  -0.971626633  -0.833646894
  C   -1.181667714  -1.691395444  -0.357522161
  H   -0.491333361  -1.910252435  -1.202732474
  H   -0.705531686  -0.945034806   0.320958726
  C   -1.690823153  -2.920718478   0.382674190
  H   -2.159812403  -2.664901829   1.362013936
  O   -2.795416709  -3.420111246  -0.459502885
  C   -2.792098320  -4.889931830  -0.490649731
  H   -3.825483536  -5.152448223  -0.152196215
  C   -0.722845340  -4.122015343   0.505761341
  H    0.028727032  -4.148282249  -0.331154304
  C   -1.680295939  -5.330037858   0.454711916
  H   -1.160456945  -6.262747060   0.156306429
  H   -2.062083944  -5.543059355   1.480787396
  O   -0.104776955  -4.177826382   1.781614061
  O   -2.344381289  -2.422417894  -3.073038440
  O   -4.539218921  -2.430696723  -1.561391078
  N   -2.629573560  -5.325054401  -1.906847345
  C   -1.369266099  -5.749021141  -2.449217749
  N   -1.393626651  -6.253311299  -3.767837206
  C   -2.517376570  -6.205285971  -4.649609227
  C   -3.744495902  -5.663308331  -4.040893102
  C   -3.771938079  -5.268883386  -2.742367951
  O   -0.329203563  -5.710680739  -1.814863329
  H   -0.482476935  -6.609805288  -4.133121460
  O   -2.353414834  -6.577650529  -5.798745591
  C   -4.944577985  -5.548466359  -4.914227144
  H   -5.322362898  -4.516162461  -4.957451509
  H   -4.730706144  -5.858983452  -5.950655623
  H   -5.770579179  -6.188375015  -4.563211140
  H   -4.686405279  -4.858725800  -2.282047408
  P    1.215250566  -3.204635570   2.089874089
  O    2.526397412  -4.138351754   1.776067963
  C    2.568903587  -5.091425041   0.705379110
  H    3.268920485  -5.863503984   1.109609180
  H    1.597856914  -5.576502257   0.511038666
  C    3.200112608  -4.406931667  -0.513716085
  H    3.949937704  -3.637598421  -0.194654453
  O    3.995265185  -5.422904645  -1.195959400
  C    3.598025452  -5.563358666  -2.567283455
  H    4.553017624  -5.685048002  -3.133245886
  C    2.221754627  -3.843893104  -1.569840010
  H    1.151856880  -4.141226712  -1.404846894
  C    2.765157189  -4.332718987  -2.915821251
  H    1.959573822  -4.525581991  -3.657833743
  H    3.397878302  -3.550618706  -3.386471723
  O    2.321495235  -2.412671776  -1.461631573
  O    1.211852473  -2.613063988   3.412435236
  O    1.067653795  -2.214331709   0.791213430
  N    2.841585743  -6.875673334  -2.643748904
  C    1.853642516  -7.151626681  -3.662799026
  N    1.264145779  -8.393456909  -3.737816401
  C    1.608259623  -9.367844729  -2.835081597
  C    2.551468052  -9.101570497  -1.777413047
  C    3.142042510  -7.869017102  -1.710302784
  O    1.539845536  -6.253437294  -4.457553745
  N    1.000605950 -10.584220951  -2.972942257
  H    1.245118940 -11.369977188  -2.389889569
  H    0.359675013 -10.756916006  -3.737888670
  H    1.059899657  -6.867913804  -6.020967736
  H    3.873758867  -7.607626134  -0.923840895
  H    2.796326212  -9.872280007  -1.047385706
  H    1.693465738  -1.989461627  -2.131462440
  H   -2.334149765  -3.726073040  -4.072694362
  H   -4.704999662  -2.485575826  -0.564826029
  H    1.835203051  -2.089942146   0.113525539
  H   -1.336411298  -8.660905236  -8.834355113
  H   -0.368401081   3.276491833  -4.676044627

