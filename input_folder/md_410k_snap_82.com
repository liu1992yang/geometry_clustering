%mem=64gb
%nproc=28       
%Chk=snap_82.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_82 

2     1 
  O    2.921705604  -4.775934229  -7.521503571
  C    2.657293108  -3.534647218  -8.188321339
  H    3.186979002  -3.672862801  -9.160972190
  H    3.131700711  -2.713086736  -7.617825401
  C    1.175655716  -3.247265508  -8.434223916
  H    1.052613914  -2.213853478  -8.848213030
  O    0.727539082  -4.126594458  -9.524385458
  C   -0.502751493  -4.758085585  -9.199411763
  H   -1.107376110  -4.790761106 -10.135847165
  C    0.174643278  -3.466803671  -7.270551647
  H    0.563554711  -4.146644998  -6.473343516
  C   -1.075319534  -4.024477483  -7.983291714
  H   -1.753481911  -3.199386870  -8.289232216
  H   -1.706201548  -4.661424835  -7.325618401
  O   -0.055262298  -2.155534351  -6.735724614
  N   -0.183263393  -6.183539133  -8.797647679
  C   -1.132361182  -7.222238628  -8.658605811
  C   -0.558728441  -8.179004456  -7.769595592
  N    0.725817587  -7.726823723  -7.416385015
  C    0.932758181  -6.533586109  -8.014793866
  N   -2.352218193  -7.347538256  -9.263292561
  C   -3.061645155  -8.460037250  -8.887975863
  N   -2.588372952  -9.425637380  -7.962032357
  C   -1.303925922  -9.312297150  -7.296371187
  N   -4.316114720  -8.625904127  -9.426860077
  H   -4.631336655  -7.976241147 -10.144901673
  H   -4.834697945  -9.487669701  -9.340319215
  O   -1.015542931 -10.130971599  -6.449548407
  H    1.885733788  -5.868807713  -7.943895017
  H   -3.189001791 -10.205608470  -7.661556875
  P   -0.807007793  -2.084577551  -5.252593402
  O   -1.748947391  -0.764332447  -5.606864925
  C   -1.200475094   0.527096209  -5.324724028
  H   -1.622097532   1.151233030  -6.141437639
  H   -0.095017155   0.554429967  -5.380978829
  C   -1.715590100   0.984075626  -3.947458933
  H   -1.009957717   0.742423674  -3.114673900
  O   -1.717328805   2.449360038  -3.956238141
  C   -3.033444713   2.976361866  -3.767835814
  H   -2.931329276   3.738865150  -2.951894328
  C   -3.170597085   0.528737152  -3.665535416
  H   -3.570625042  -0.150720749  -4.460316383
  C   -3.990749909   1.820327720  -3.496470294
  H   -4.384449256   1.856123342  -2.447400267
  H   -4.893006672   1.806225793  -4.134758552
  O   -3.286600305  -0.098744138  -2.380886275
  O    0.171852862  -1.974780699  -4.153937678
  O   -1.834835758  -3.327317967  -5.468183901
  N   -3.376492001   3.705813628  -5.039547304
  C   -2.490907735   4.145661053  -6.014535505
  C   -3.267230907   4.834021199  -7.004026898
  N   -4.617022315   4.822639957  -6.627182972
  C   -4.678369878   4.164794631  -5.469320764
  N   -1.106032919   4.026896565  -6.130315939
  C   -0.526648141   4.634213701  -7.259067799
  N   -1.203168130   5.270148776  -8.201983200
  C   -2.607015614   5.414252597  -8.134383625
  H    0.579320507   4.562734088  -7.359557933
  N   -3.228661453   6.075104422  -9.122857457
  H   -2.710135773   6.477960691  -9.904468921
  H   -4.241135407   6.211158198  -9.109432317
  H   -5.573489913   3.991245234  -4.875979440
  P   -2.656258903  -1.618203755  -2.191785617
  O   -1.453614138  -1.134571069  -1.204589814
  C   -0.323139303  -2.047261357  -1.106828753
  H    0.052097885  -2.309437134  -2.125945172
  H    0.441877182  -1.428883940  -0.589689618
  C   -0.830069283  -3.214604105  -0.267545584
  H   -1.043622320  -2.911373101   0.786299456
  O   -2.144619121  -3.497920435  -0.868570870
  C   -2.345580313  -4.939377524  -1.034600976
  H   -3.356814171  -5.106248882  -0.586349907
  C   -0.063143548  -4.551957925  -0.308462597
  H    0.587396573  -4.654684882  -1.225711962
  C   -1.191985937  -5.603445639  -0.286849325
  H   -0.868890832  -6.576722408  -0.722578649
  H   -1.487323641  -5.863139841   0.753646120
  O    0.769872150  -4.539695716   0.856132061
  O   -2.297152499  -2.244553041  -3.501013935
  O   -3.953443200  -2.262419868  -1.450973635
  N   -2.411099528  -5.245167896  -2.490815649
  C   -1.263145790  -5.684558091  -3.228256070
  N   -1.485336082  -6.130662180  -4.545525522
  C   -2.745406773  -6.078392757  -5.225813547
  C   -3.834335971  -5.463481607  -4.444400315
  C   -3.655533099  -5.090108127  -3.153180517
  O   -0.145459432  -5.710477532  -2.742941360
  H   -0.642311608  -6.476170307  -5.045760587
  O   -2.784539943  -6.519846334  -6.355741731
  C   -5.114124057  -5.249231315  -5.173258580
  H   -5.366846792  -6.116102624  -5.808298115
  H   -5.971438860  -5.082201367  -4.507490207
  H   -5.041731927  -4.378746038  -5.847368592
  H   -4.460434767  -4.629167098  -2.559944054
  P    1.710432678  -5.892212768   1.096569499
  O    2.860100517  -5.784052598  -0.057119559
  C    4.043578053  -4.970288820   0.165149215
  H    4.003593821  -4.397545016   1.115271930
  H    4.871848109  -5.712121516   0.220683214
  C    4.217542326  -4.044897077  -1.042586137
  H    5.147255309  -3.433996067  -0.917395750
  O    4.501047698  -4.834393526  -2.232829954
  C    3.436094441  -4.745398573  -3.187374387
  H    3.930506358  -4.452063676  -4.145576609
  C    2.981018079  -3.166654622  -1.366984029
  H    2.268309223  -3.108763224  -0.512739551
  C    2.393164566  -3.762659255  -2.651005576
  H    1.418284521  -4.280178222  -2.454475840
  H    2.110271152  -2.984825142  -3.397080022
  O    3.343408855  -1.802531698  -1.511718274
  O    0.983852363  -7.159377048   1.100065208
  O    2.498856124  -5.361063044   2.423416863
  N    2.867159797  -6.129584768  -3.384692277
  C    1.974911783  -6.339091493  -4.503875022
  N    1.439432677  -7.591018635  -4.716591864
  C    1.639223660  -8.597243665  -3.793362687
  C    2.481275484  -8.384560378  -2.650569545
  C    3.076453577  -7.160667828  -2.472802945
  O    1.726331198  -5.386563378  -5.250950871
  N    1.004658302  -9.776464758  -4.053382150
  H    1.091459991 -10.575743024  -3.442287930
  H    0.396797595  -9.881858403  -4.868254532
  H    2.688140879  -4.731985011  -6.541224633
  H    3.751215220  -6.945287240  -1.624080930
  H    2.640594120  -9.180443740  -1.922543975
  H    3.926342279  -1.649792197  -2.287459647
  H   -2.467001670  -3.506885018  -4.673334695
  H   -3.929848689  -2.325261225  -0.447707683
  H    2.143416702  -5.606310340   3.323107786
  H    1.280801464  -8.127126038  -6.647130369
  H   -0.575845603   3.552475346  -5.381678731
