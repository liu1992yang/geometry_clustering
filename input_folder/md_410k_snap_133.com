%mem=64gb
%nproc=28       
%Chk=snap_133.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_133 

2     1 
  O    0.991031823  -2.425857461  -5.792726662
  C    1.164262133  -3.227588519  -6.952639782
  H    1.433024815  -4.271106985  -6.666848903
  H    2.036261087  -2.768940711  -7.463611260
  C   -0.084420264  -3.177791561  -7.848912880
  H   -0.244315606  -2.167311720  -8.286508605
  O    0.194806425  -4.052598476  -8.996907341
  C   -0.857207417  -4.990107820  -9.199494238
  H   -1.012413243  -5.067292964 -10.302377094
  C   -1.389422441  -3.725781005  -7.217417287
  H   -1.174211957  -4.339874841  -6.315151423
  C   -2.040364224  -4.558170889  -8.334227676
  H   -2.766595643  -3.935543939  -8.902880453
  H   -2.640813048  -5.405028235  -7.936457293
  O   -2.256666737  -2.623981003  -6.933589049
  N   -0.334205265  -6.328428264  -8.716017416
  C   -0.942389931  -7.582899125  -8.927984130
  C   -0.295443007  -8.504337903  -8.051550657
  N    0.667280326  -7.801760758  -7.317779466
  C    0.627803988  -6.504796784  -7.703562297
  N   -1.940811114  -7.904881805  -9.806028610
  C   -2.321485347  -9.224170714  -9.779991047
  N   -1.757622751 -10.190863496  -8.908419716
  C   -0.707258175  -9.881796299  -7.951372072
  N   -3.320262496  -9.607172537 -10.637882489
  H   -3.705364777  -8.926987307 -11.291168525
  H   -3.621439924 -10.564689054 -10.740897302
  O   -0.311048168 -10.754616180  -7.215955852
  H    1.263391631  -5.687724184  -7.302984614
  H   -2.105815717 -11.159956629  -8.903315464
  P   -2.539986827  -2.356777473  -5.314691903
  O   -2.790406427  -0.730379199  -5.421482090
  C   -1.662194581   0.146445721  -5.280612612
  H   -1.815873654   0.887293794  -6.093312019
  H   -0.688155707  -0.363990193  -5.428440260
  C   -1.772421398   0.804348762  -3.894419943
  H   -1.034606894   0.385184646  -3.157334859
  O   -1.328698885   2.193510613  -4.025698084
  C   -2.328660764   3.111869347  -3.569547515
  H   -1.814373840   3.763006759  -2.814862808
  C   -3.225790911   0.840269604  -3.349733369
  H   -3.969017168   0.379001050  -4.049360763
  C   -3.527174627   2.320743716  -3.059456233
  H   -3.662839903   2.446451863  -1.953745755
  H   -4.498550367   2.620228927  -3.494767925
  O   -3.350471211   0.214537340  -2.062871535
  O   -1.381144430  -2.902436965  -4.551213664
  O   -4.061566671  -2.907087686  -5.270427575
  N   -2.659280522   3.968185178  -4.761849368
  C   -1.856108557   4.171333426  -5.876977282
  C   -2.512316096   5.151986295  -6.691239099
  N   -3.707548214   5.545252091  -6.074025566
  C   -3.788302023   4.853614239  -4.937115073
  N   -0.631392056   3.617920397  -6.250267499
  C   -0.075723911   4.099796446  -7.448416623
  N   -0.642648336   5.005352509  -8.230087703
  C   -1.887832131   5.587200878  -7.903766561
  H    0.908384543   3.680750930  -7.755234805
  N   -2.400552058   6.504860141  -8.737824910
  H   -1.910233833   6.789889459  -9.586124786
  H   -3.295671915   6.954817186  -8.539357218
  H   -4.579382166   4.928799096  -4.195008254
  P   -2.944234321  -1.388013599  -1.947229855
  O   -1.365588343  -1.034644797  -1.684096730
  C   -0.387985127  -2.089722490  -1.544083995
  H   -0.309420502  -2.651972727  -2.505668098
  H    0.558493986  -1.535537544  -1.353161109
  C   -0.853924389  -2.936571973  -0.364708918
  H   -0.989690006  -2.350893268   0.576910464
  O   -2.213749692  -3.320702370  -0.788170544
  C   -2.292642230  -4.767207343  -0.998389041
  H   -3.322961092  -5.005827664  -0.633761328
  C   -0.091672807  -4.245660299  -0.090101155
  H    0.739409173  -4.402555087  -0.837686735
  C   -1.161134392  -5.351476692  -0.156406648
  H   -0.751598408  -6.308197838  -0.545056299
  H   -1.519600676  -5.609985650   0.866907007
  O    0.457477037  -4.105819398   1.227852209
  O   -3.304271085  -2.135339942  -3.196237762
  O   -3.846054406  -1.675101060  -0.640889018
  N   -2.197529211  -5.100308113  -2.443269746
  C   -0.950043745  -5.273011152  -3.127172135
  N   -1.009080012  -5.931853694  -4.375139930
  C   -2.222111573  -6.236431565  -5.073725084
  C   -3.449328603  -5.769123436  -4.413781827
  C   -3.410457659  -5.225537646  -3.172425895
  O    0.131033958  -4.956635530  -2.656111362
  H   -0.101606170  -6.171691384  -4.821619567
  O   -2.115259774  -6.800668403  -6.146850985
  C   -4.702684696  -5.883373301  -5.208282793
  H   -5.593831467  -6.043101504  -4.586665759
  H   -4.876138201  -4.966531603  -5.804142025
  H   -4.656222363  -6.722407768  -5.922614518
  H   -4.313337057  -4.856184998  -2.668058991
  P    1.646139726  -5.219522518   1.583443243
  O    2.925714085  -4.534452763   0.795254556
  C    3.556201418  -5.337684390  -0.230644786
  H    4.440525959  -5.819681650   0.239368987
  H    2.886940518  -6.123665017  -0.634727824
  C    3.990662754  -4.319424263  -1.296787443
  H    4.737023159  -3.602599545  -0.884688198
  O    4.726027039  -5.072065795  -2.308855886
  C    4.020997043  -5.100759906  -3.553803817
  H    4.812177896  -4.987746922  -4.334053337
  C    2.825653739  -3.611235280  -2.031123587
  H    1.824276676  -3.900489059  -1.613520758
  C    2.957330555  -4.009496935  -3.506980267
  H    1.965043855  -4.333181694  -3.907985561
  H    3.214840397  -3.148714928  -4.155501794
  O    2.828444181  -2.208511685  -1.836641233
  O    1.347701633  -6.605096211   1.217257344
  O    1.925955127  -4.813576033   3.121309687
  N    3.436294791  -6.497697788  -3.694312695
  C    2.408169202  -6.764706053  -4.671574501
  N    1.956845856  -8.060565586  -4.842722881
  C    2.462413573  -9.090594149  -4.085581376
  C    3.478130654  -8.831280338  -3.097885919
  C    3.948531757  -7.554237949  -2.940777317
  O    1.943283064  -5.848136400  -5.352203872
  N    1.965006597 -10.333367710  -4.344431672
  H    2.269767603 -11.148583492  -3.830951002
  H    1.249731319 -10.476795015  -5.053359360
  H    0.325054687  -2.842434323  -5.159623347
  H    4.751068418  -7.305346663  -2.222110882
  H    3.874703477  -9.645289774  -2.489539516
  H    3.649587184  -1.785141256  -2.171217358
  H   -4.573827180  -2.710640737  -4.401315923
  H   -3.494474189  -2.288182559   0.084544449
  H    2.199447184  -3.866178181   3.314668913
  H    1.227736604  -8.202042651  -6.526106522
  H   -0.183105945   2.921971330  -5.632102112

