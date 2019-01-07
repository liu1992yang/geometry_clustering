%mem=64gb
%nproc=28       
%Chk=snap_72.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_72 

2     1 
  O    2.634967954  -6.018740035  -6.860104981
  C    2.899503369  -4.681145666  -7.299666433
  H    3.417759721  -4.844963462  -8.274346476
  H    3.604129543  -4.208131244  -6.590489373
  C    1.654673988  -3.815244204  -7.512140239
  H    1.939035325  -2.731692652  -7.584648077
  O    1.131836212  -4.104266019  -8.845817447
  C   -0.198156008  -4.598384510  -8.790137628
  H   -0.743112813  -4.111172915  -9.638792588
  C    0.480327072  -3.987518326  -6.515596391
  H    0.680376467  -4.737047430  -5.710212507
  C   -0.735633917  -4.345358595  -7.383515996
  H   -1.490139464  -3.526567131  -7.396180518
  H   -1.307977346  -5.209363495  -6.964931801
  O    0.409671276  -2.686520360  -5.896615091
  N   -0.128094890  -6.074270314  -9.098414641
  C   -0.832756113  -6.752815663 -10.117267325
  C   -0.437748638  -8.118735582 -10.042445882
  N    0.496992287  -8.241357007  -8.995388576
  C    0.678073595  -7.015846269  -8.443223609
  N   -1.714370659  -6.227537457 -11.020372482
  C   -2.226888795  -7.138445071 -11.916290254
  N   -1.885455928  -8.509153296 -11.913997838
  C   -0.961892485  -9.109288876 -10.953021285
  N   -3.120504760  -6.674825816 -12.847058805
  H   -3.353666009  -5.684867897 -12.870858052
  H   -3.509612916  -7.257458975 -13.574343399
  O   -0.753010441 -10.290942534 -11.033097097
  H    1.420397957  -6.717217530  -7.608729159
  H   -2.292811765  -9.157181987 -12.605623243
  P   -0.728459816  -2.346282675  -4.744559542
  O   -1.342142835  -0.960597123  -5.416542449
  C   -0.904541863   0.282866156  -4.849294909
  H   -1.038071624   0.988830996  -5.696841820
  H    0.163717324   0.265388701  -4.558875104
  C   -1.814056081   0.647075653  -3.660639237
  H   -1.419117010   0.277538565  -2.680520281
  O   -1.749033355   2.093684603  -3.471828619
  C   -2.986226458   2.724873269  -3.816257914
  H   -3.207043361   3.419365867  -2.963059614
  C   -3.300141098   0.292019066  -3.928103903
  H   -3.429939101  -0.376746904  -4.817112903
  C   -4.026213427   1.641685277  -4.083778971
  H   -4.858884761   1.685014674  -3.335149984
  H   -4.523227621   1.713163991  -5.068343402
  O   -3.950559329  -0.316199916  -2.800269880
  O   -0.055650982  -2.144700050  -3.438241483
  O   -1.806370766  -3.479567106  -5.155700133
  N   -2.671723614   3.560816853  -5.028416831
  C   -1.413862180   4.053449317  -5.360020522
  C   -1.585302551   4.907844295  -6.496922417
  N   -2.939519318   4.935142760  -6.857180462
  C   -3.576312238   4.147637924  -5.989184036
  N   -0.158372713   3.839292998  -4.791415192
  C    0.914309128   4.527450058  -5.384684991
  N    0.803784646   5.329850850  -6.431315817
  C   -0.445401114   5.585001945  -7.038982889
  H    1.922972694   4.375121941  -4.939098864
  N   -0.485216613   6.443560275  -8.069121841
  H    0.357082363   6.912347409  -8.405714172
  H   -1.365625485   6.666221548  -8.535643107
  H   -4.645708387   3.953365861  -5.967679656
  P   -3.284942329  -1.714967748  -2.208364453
  O   -2.424701013  -0.991232181  -1.032519949
  C   -1.245784710  -1.718406416  -0.583513767
  H   -0.579394809  -1.945119967  -1.447550893
  H   -0.745262561  -0.981525711   0.084633634
  C   -1.743674930  -2.948928154   0.159830298
  H   -2.157341994  -2.710083752   1.167071277
  O   -2.896006587  -3.410839476  -0.636457006
  C   -2.901477126  -4.881538736  -0.715858776
  H   -3.921943495  -5.148796107  -0.343189523
  C   -0.781531989  -4.160938077   0.194124220
  H   -0.088598721  -4.173686486  -0.693741616
  C   -1.754449871  -5.356146863   0.172927398
  H   -1.262059869  -6.288712656  -0.169203053
  H   -2.095729836  -5.583555939   1.208632679
  O   -0.070927944  -4.237000367   1.421708580
  O   -2.531010578  -2.413333166  -3.298322402
  O   -4.663138538  -2.407112599  -1.707446154
  N   -2.794401428  -5.268419099  -2.147481635
  C   -1.561469374  -5.688012518  -2.750230753
  N   -1.643597669  -6.173099956  -4.066220799
  C   -2.819154453  -6.161534418  -4.889913776
  C   -4.014683859  -5.615159510  -4.222141370
  C   -3.975933976  -5.206918209  -2.930098079
  O   -0.492995640  -5.655622394  -2.157249269
  H   -0.759605813  -6.587306447  -4.460263312
  O   -2.700550373  -6.550668171  -6.033018928
  C   -5.264077153  -5.545789604  -5.028289595
  H   -5.706306471  -4.539129464  -5.027492875
  H   -5.087719335  -5.819094295  -6.083432087
  H   -6.027955141  -6.245565321  -4.650991644
  H   -4.865138634  -4.791993304  -2.428024922
  P    1.306533210  -3.320202870   1.611279068
  O    2.560909433  -4.299194209   1.210429428
  C    2.543912151  -5.209434645   0.103392126
  H    3.257708178  -5.999178238   0.440417146
  H    1.559287919  -5.680391485  -0.056660052
  C    3.114194971  -4.475139536  -1.118436382
  H    3.862183433  -3.701590310  -0.804247077
  O    3.910756872  -5.425842289  -1.884193527
  C    3.369370596  -5.626058405  -3.195885174
  H    4.259744976  -5.709432046  -3.865243787
  C    2.060446333  -3.901816360  -2.089714274
  H    0.997077227  -4.144695534  -1.814590584
  C    2.433020664  -4.450586415  -3.474711057
  H    1.523454306  -4.723240158  -4.053047898
  H    2.948581798  -3.665923422  -4.066004548
  O    2.231659465  -2.476219842  -2.018559314
  O    1.440607943  -2.719178508   2.923298933
  O    1.100207219  -2.324960503   0.321402047
  N    2.695303079  -6.982889468  -3.145171683
  C    1.530385870  -7.348535542  -3.927617798
  N    0.922408262  -8.558224452  -3.729733298
  C    1.420682951  -9.437885405  -2.798501804
  C    2.604968359  -9.121567594  -2.039972073
  C    3.205095381  -7.907722694  -2.227650260
  O    1.120817832  -6.544629179  -4.786126008
  N    0.749692489 -10.614475984  -2.633871950
  H    1.068637867 -11.323794572  -1.990597110
  H   -0.068085288 -10.828594111  -3.193399100
  H    2.293033362  -6.046391216  -5.909929144
  H    4.107701310  -7.606033034  -1.664074743
  H    3.018412831  -9.839787011  -1.332225840
  H    1.534305015  -2.038166357  -2.610693595
  H   -2.579777852  -3.626452580  -4.468354852
  H   -4.825545864  -2.428192120  -0.709835338
  H    1.837059665  -2.171775745  -0.376624869
  H    0.933808029  -9.117945274  -8.728069768
  H   -0.076988047   3.231984460  -3.959174157

