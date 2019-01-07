%mem=64gb
%nproc=28       
%Chk=snap_120.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_120 

2     1 
  O    3.188041223  -3.391469977  -6.332083936
  C    2.871544712  -2.599814043  -7.468985016
  H    3.396508688  -2.977885969  -8.366893228
  H    3.284465840  -1.603743596  -7.197865136
  C    1.357774293  -2.520057794  -7.708752831
  H    1.057881018  -1.549730767  -8.173492565
  O    1.009815265  -3.506457683  -8.738566089
  C    0.007123692  -4.401722938  -8.284281225
  H   -0.814908020  -4.379028683  -9.045842785
  C    0.497301375  -2.834649707  -6.466922786
  H    1.132220099  -3.022544849  -5.553801748
  C   -0.384704461  -4.030339136  -6.853804695
  H   -1.472099937  -3.780814658  -6.808467423
  H   -0.248462099  -4.870524433  -6.140632451
  O   -0.294430674  -1.640181626  -6.263383209
  N    0.609607603  -5.787144879  -8.346763918
  C   -0.041346312  -6.968912818  -7.937288099
  C    0.866584603  -8.038699204  -8.165178395
  N    2.056756334  -7.493574604  -8.676293011
  C    1.908572185  -6.139108922  -8.742235825
  N   -1.309517116  -7.089973592  -7.435144050
  C   -1.704935116  -8.385239755  -7.190196858
  N   -0.847463545  -9.500101088  -7.347967361
  C    0.535281882  -9.396867357  -7.800045015
  N   -2.975206857  -8.564002710  -6.710910667
  H   -3.606735009  -7.743094813  -6.667851305
  H   -3.405617043  -9.474698108  -6.636000213
  O    1.197713845 -10.405310659  -7.825122489
  H    2.648368584  -5.412540513  -9.087802836
  H   -1.171652244 -10.447401943  -7.103851305
  P   -1.146021525  -1.631081073  -4.838800931
  O   -1.340296874   0.021240727  -4.798781409
  C   -0.558975913   0.696525241  -3.777293577
  H    0.092670899   1.406897459  -4.334318560
  H    0.089411541  -0.001846676  -3.202023734
  C   -1.525961197   1.445196939  -2.855796087
  H   -1.239222458   1.325734493  -1.777271357
  O   -1.329638120   2.866003914  -3.114971435
  C   -2.577706879   3.549348739  -3.264366011
  H   -2.561809881   4.385865470  -2.518223839
  C   -3.036383970   1.156202817  -3.077692629
  H   -3.255321013   0.587841722  -4.014761148
  C   -3.707509765   2.536184878  -3.075936976
  H   -4.229640826   2.684119501  -2.096574764
  H   -4.516686787   2.622411675  -3.824357804
  O   -3.567920887   0.445105585  -1.952216115
  O   -0.355891817  -2.205607040  -3.727596061
  O   -2.508542067  -2.258534135  -5.453943882
  N   -2.539006644   4.164716442  -4.628225388
  C   -2.249821695   3.577828898  -5.859175512
  C   -2.082681946   4.649264109  -6.798334018
  N   -2.264229740   5.880013377  -6.148799803
  C   -2.517442680   5.594781517  -4.875181091
  N   -2.142506143   2.253405928  -6.275315778
  C   -1.800823415   2.042402154  -7.622623249
  N   -1.617854487   3.001634312  -8.517130054
  C   -1.739420519   4.361076476  -8.157295073
  H   -1.689426831   0.981407096  -7.944537060
  N   -1.511333002   5.296574679  -9.093744728
  H   -1.257712771   5.048342340 -10.049234590
  H   -1.591357298   6.290185819  -8.870801135
  H   -2.709365927   6.310857755  -4.077820130
  P   -3.157985802  -1.167281967  -1.890761418
  O   -1.947413148  -0.972584513  -0.821807505
  C   -0.987669481  -2.062742619  -0.713129360
  H   -0.530575291  -2.290359805  -1.709278176
  H   -0.209056654  -1.633881889  -0.047644269
  C   -1.729077682  -3.235785621  -0.087444006
  H   -1.968985289  -3.080340376   0.991028609
  O   -3.029171699  -3.215988691  -0.778331314
  C   -3.417718533  -4.562721705  -1.204924755
  H   -4.452476186  -4.661974774  -0.788008599
  C   -1.147686847  -4.640516910  -0.342204585
  H   -0.444518254  -4.659013667  -1.224382162
  C   -2.395340460  -5.512147814  -0.593978491
  H   -2.149950576  -6.403361092  -1.217415834
  H   -2.775702996  -5.948861495   0.356813045
  O   -0.453930818  -4.991351029   0.859024639
  O   -2.797449498  -1.655752359  -3.264295104
  O   -4.588848545  -1.675268535  -1.313536649
  N   -3.496709367  -4.584677513  -2.690636536
  C   -2.456481943  -5.131492801  -3.511655026
  N   -2.861170259  -5.523090437  -4.812717893
  C   -4.186281525  -5.366853227  -5.347846948
  C   -5.050105295  -4.476347832  -4.554655017
  C   -4.713697247  -4.151111999  -3.283666838
  O   -1.310133130  -5.294780123  -3.143626305
  H   -2.178816030  -6.051038580  -5.383364296
  O   -4.447304296  -5.985570958  -6.358827279
  C   -6.303382427  -4.018144903  -5.209849659
  H   -6.096346171  -3.492730667  -6.156739997
  H   -6.949532466  -4.877491626  -5.469718576
  H   -6.904420176  -3.344249736  -4.584040115
  H   -5.362440559  -3.529563681  -2.643749648
  P    0.419386912  -6.406241907   0.780988232
  O    1.955358497  -5.877220573   0.530566176
  C    2.598241827  -6.299507097  -0.696360630
  H    3.553756676  -6.755277784  -0.353111912
  H    2.020680632  -7.058180253  -1.262625529
  C    2.867762472  -5.013956956  -1.487644086
  H    3.280620494  -4.206216172  -0.830313032
  O    3.952283373  -5.310295665  -2.417113050
  C    3.518223671  -5.233190253  -3.776222269
  H    4.282532520  -4.602363549  -4.294378693
  C    1.684102292  -4.494564976  -2.330269312
  H    0.699632294  -4.985047474  -2.103275373
  C    2.085247833  -4.692707287  -3.795716473
  H    1.374416886  -5.371985456  -4.308166195
  H    2.033016391  -3.735720368  -4.372354720
  O    1.551782560  -3.122389733  -1.954017632
  O   -0.097286044  -7.384943046  -0.183102616
  O    0.469491444  -6.754054537   2.360492039
  N    3.610568212  -6.640139992  -4.338424091
  C    3.035022074  -6.944064875  -5.634991758
  N    3.098321686  -8.227098697  -6.121888208
  C    3.708409306  -9.217455254  -5.384800020
  C    4.352233412  -8.915784411  -4.132663253
  C    4.288348664  -7.637705255  -3.642438894
  O    2.494222357  -6.027344392  -6.264202582
  N    3.664555148 -10.473676194  -5.913140089
  H    4.090924351 -11.264172321  -5.453340101
  H    3.125326279 -10.659350234  -6.759245009
  H    3.023010629  -4.364697789  -6.501433158
  H    4.763857651  -7.345815320  -2.688906200
  H    4.885269149  -9.693558443  -3.586106929
  H    1.107962437  -2.599339941  -2.676652737
  H   -3.230108733  -2.514962070  -4.758548306
  H   -4.651168992  -1.845767004  -0.320375038
  H    1.025734559  -6.162006908   2.952204158
  H    2.925363313  -8.024175539  -8.737431230
  H   -2.185744753   1.458934359  -5.605976019

