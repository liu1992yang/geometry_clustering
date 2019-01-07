%mem=64gb
%nproc=28       
%Chk=snap_117.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_117 

2     1 
  O    3.028662024  -4.715158070  -7.259243258
  C    2.928053595  -3.294729292  -7.277660624
  H    3.667304451  -3.014281507  -8.062054469
  H    3.261770279  -2.877960850  -6.308521448
  C    1.532796185  -2.781362469  -7.653518800
  H    1.551028382  -1.678710650  -7.848674687
  O    1.170545872  -3.344744007  -8.955645630
  C    0.076352469  -4.239327364  -8.825905345
  H   -0.497753610  -4.167295078  -9.782304008
  C    0.375438460  -3.132937546  -6.683044028
  H    0.699644293  -3.724502737  -5.787036689
  C   -0.660184802  -3.878576048  -7.539088229
  H   -1.543857751  -3.232900123  -7.755915395
  H   -1.098461904  -4.755755936  -7.016604529
  O   -0.122428567  -1.834627634  -6.283325786
  N    0.679701324  -5.631616767  -8.772289133
  C    0.091193360  -6.812165764  -8.269336906
  C    1.057214513  -7.848701691  -8.418482074
  N    2.228628915  -7.274241227  -8.942198487
  C    2.000348269  -5.944981074  -9.137572756
  N   -1.168974714  -6.965852251  -7.772522350
  C   -1.494780978  -8.248519594  -7.400025114
  N   -0.606391982  -9.339554275  -7.528868444
  C    0.771503921  -9.192218619  -7.975646041
  N   -2.740238435  -8.443264826  -6.861961875
  H   -3.390530083  -7.639687424  -6.853780790
  H   -3.136711570  -9.360708068  -6.722122016
  O    1.469497163 -10.180309711  -7.957320551
  H    2.713222255  -5.197527914  -9.510477251
  H   -0.876893108 -10.272170888  -7.190590434
  P   -1.071868972  -1.794409155  -4.926656613
  O   -1.197220685  -0.130512491  -4.897015493
  C   -0.492516620   0.512536138  -3.806860881
  H    0.200671335   1.231497272  -4.300156785
  H    0.113838900  -0.200349000  -3.208440557
  C   -1.510385162   1.248062117  -2.928739391
  H   -1.302142658   1.087823081  -1.836672673
  O   -1.261131892   2.670693576  -3.127563213
  C   -2.482023876   3.396462896  -3.299880655
  H   -2.493414942   4.178822386  -2.497735406
  C   -3.008602408   1.006469644  -3.261392436
  H   -3.174781801   0.468176060  -4.225351283
  C   -3.644666381   2.403499085  -3.256540166
  H   -4.247483862   2.521775505  -2.320008063
  H   -4.383985209   2.541420923  -4.066959593
  O   -3.654971607   0.290039428  -2.199769513
  O   -0.379615419  -2.402270278  -3.769340335
  O   -2.438300873  -2.304352000  -5.628840176
  N   -2.357821484   4.100363432  -4.612293997
  C   -2.062473878   3.584791262  -5.871484692
  C   -1.957811416   4.706813313  -6.760147888
  N   -2.170904631   5.896131483  -6.047032204
  C   -2.393475080   5.539542981  -4.785113624
  N   -1.891364756   2.285588438  -6.345278524
  C   -1.580337385   2.152969004  -7.710032945
  N   -1.478534703   3.158393695  -8.564830053
  C   -1.658187991   4.493876131  -8.143185928
  H   -1.422620059   1.116466659  -8.085803728
  N   -1.536040038   5.469921795  -9.056854718
  H   -1.317162926   5.263106189 -10.031439803
  H   -1.656486663   6.450643099  -8.799296596
  H   -2.591823757   6.207986186  -3.948828182
  P   -3.230901065  -1.311450088  -2.065681966
  O   -2.082660686  -1.045706408  -0.939302012
  C   -1.116125521  -2.108362835  -0.727220909
  H   -0.616383638  -2.371439447  -1.694558908
  H   -0.374274285  -1.641251765  -0.043844809
  C   -1.865165212  -3.265955779  -0.084455922
  H   -2.166294577  -3.062789328   0.970854361
  O   -3.121070283  -3.314996364  -0.847758437
  C   -3.434254356  -4.693013009  -1.253179786
  H   -4.493651977  -4.814930440  -0.911946321
  C   -1.228679216  -4.663346333  -0.229557869
  H   -0.454802574  -4.701944187  -1.049684167
  C   -2.431057117  -5.581204617  -0.529032554
  H   -2.124730619  -6.490306974  -1.093796856
  H   -2.861411403  -5.982708356   0.416261110
  O   -0.642961917  -4.943495485   1.043403869
  O   -2.776850983  -1.836662140  -3.396484688
  O   -4.678314232  -1.832186978  -1.552061307
  N   -3.395932028  -4.754220800  -2.736653346
  C   -2.241983790  -5.214798423  -3.461657222
  N   -2.459215506  -5.486018604  -4.834185155
  C   -3.698396908  -5.266627285  -5.527192890
  C   -4.697070758  -4.499474636  -4.766731408
  C   -4.545879819  -4.303065274  -3.435123388
  O   -1.163796360  -5.416274341  -2.941998677
  H   -1.682504662  -5.936556829  -5.355757612
  O   -3.810609672  -5.733342106  -6.642023895
  C   -5.872373156  -4.017721275  -5.541612896
  H   -6.574022002  -3.416641484  -4.948441431
  H   -5.560403156  -3.406525917  -6.404690277
  H   -6.446519481  -4.864208905  -5.962343827
  H   -5.307603458  -3.784302335  -2.829471026
  P    0.446886828  -6.204722520   1.055589246
  O    1.822979959  -5.414430858   0.615118025
  C    2.534485250  -5.957153442  -0.523206209
  H    3.470423090  -6.390635360  -0.105380110
  H    1.969246727  -6.749522373  -1.056989266
  C    2.854685831  -4.750058297  -1.414386294
  H    3.399204328  -3.951767204  -0.853670475
  O    3.815477255  -5.233727929  -2.407310969
  C    3.255276459  -5.210969186  -3.726035207
  H    4.113050727  -4.942207660  -4.389926072
  C    1.667184253  -4.169422631  -2.210453428
  H    0.695335007  -4.702837414  -2.036040892
  C    2.099677970  -4.217772436  -3.686468463
  H    1.238411144  -4.472273476  -4.348303112
  H    2.443496449  -3.211401800  -4.000664123
  O    1.522338370  -2.818664747  -1.765130299
  O    0.091244165  -7.372899826   0.247489470
  O    0.615571768  -6.344588157   2.658147206
  N    2.842541100  -6.632990202  -4.049371568
  C    1.825714815  -6.969714393  -5.042317236
  N    1.624258357  -8.289402292  -5.366588029
  C    2.322107369  -9.282081154  -4.718908776
  C    3.275835740  -8.971321339  -3.686045296
  C    3.526170696  -7.659468456  -3.395435697
  O    1.232275791  -6.041688866  -5.603801449
  N    2.068936870 -10.572494348  -5.096458047
  H    2.624446484 -11.338227644  -4.741344469
  H    1.536818750 -10.756105216  -5.943468913
  H    2.468776177  -5.126518675  -6.539225333
  H    4.282443248  -7.359621729  -2.646379732
  H    3.796369135  -9.767826512  -3.154550756
  H    0.880535388  -2.344854652  -2.369358535
  H   -3.211523349  -2.532789656  -4.976517238
  H   -4.769603584  -2.055946656  -0.568726792
  H    0.848188898  -5.518496049   3.179362524
  H    3.088855594  -7.788742091  -9.106383720
  H   -1.887379971   1.464076547  -5.711004482
