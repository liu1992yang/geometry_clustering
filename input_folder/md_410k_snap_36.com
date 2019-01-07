%mem=64gb
%nproc=28       
%Chk=snap_36.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_36 

2     1 
  O    1.968103332  -3.825281078  -6.247264213
  C    2.272515551  -3.594255584  -7.616443185
  H    2.471658249  -4.606613503  -8.021414808
  H    3.206197566  -3.001295808  -7.686159684
  C    1.116481342  -2.907188572  -8.358573298
  H    1.308399327  -1.821851979  -8.530973616
  O    1.088689453  -3.481229490  -9.705941961
  C   -0.191374850  -4.000786938 -10.022423960
  H   -0.397412738  -3.685613352 -11.078736587
  C   -0.295880505  -3.127420564  -7.759630353
  H   -0.305897118  -3.899490419  -6.957388091
  C   -1.176270341  -3.546162619  -8.950376468
  H   -1.770676805  -2.667106907  -9.292765964
  H   -1.934309196  -4.300722991  -8.664885774
  O   -0.765999209  -1.851882023  -7.307549005
  N   -0.061048229  -5.510753804 -10.041935885
  C   -0.064237732  -6.300833679 -11.213976314
  C    0.111109366  -7.652344130 -10.800341093
  N    0.229632634  -7.660370531  -9.399784520
  C    0.129613729  -6.379003467  -8.955530927
  N   -0.231316445  -5.875716036 -12.499266820
  C   -0.209788960  -6.877161725 -13.445983340
  N   -0.042478838  -8.241755263 -13.123341768
  C    0.135154173  -8.735520316 -11.758607115
  N   -0.375342131  -6.507800032 -14.753748699
  H   -0.469968557  -5.520796477 -14.989776785
  H   -0.345326018  -7.164267162 -15.521433307
  O    0.269189126  -9.918941714 -11.599939856
  H    0.150520352  -6.057141139  -7.885315548
  H   -0.040081032  -8.962012479 -13.862484888
  P   -0.917149632  -1.714611333  -5.650756769
  O   -1.599337298  -0.208362044  -5.710981941
  C   -0.695827946   0.864592052  -5.348993500
  H   -0.695309352   1.543659978  -6.230089446
  H    0.346013698   0.515469101  -5.166673661
  C   -1.270634658   1.565941217  -4.114642391
  H   -0.506416676   1.690986097  -3.306710926
  O   -1.570808329   2.947059514  -4.507741044
  C   -2.922998142   3.277680295  -4.237093879
  H   -2.896355449   4.292644854  -3.779440143
  C   -2.592660596   0.960911283  -3.567612427
  H   -3.038231503   0.182691092  -4.237785717
  C   -3.527502299   2.168107299  -3.376604884
  H   -3.535236490   2.448050438  -2.294122871
  H   -4.580393626   1.928039966  -3.611634556
  O   -2.370082720   0.436084910  -2.256851406
  O    0.411394796  -1.834095534  -5.000702179
  O   -2.095198224  -2.823973470  -5.510897108
  N   -3.618631619   3.366805486  -5.572237113
  C   -4.242888277   4.474616068  -6.137458274
  C   -4.717506582   4.067667571  -7.428288174
  N   -4.363998432   2.730165659  -7.658289251
  C   -3.710944481   2.324468911  -6.569238637
  N   -4.467938853   5.772663615  -5.673817966
  C   -5.169597092   6.636273619  -6.540619160
  N   -5.625553682   6.295575670  -7.732563925
  C   -5.429765678   4.996916692  -8.250363861
  H   -5.348805598   7.675324562  -6.179143334
  N   -5.923037285   4.715572026  -9.466892379
  H   -6.422718392   5.415561817 -10.016250301
  H   -5.807935727   3.787845153  -9.875595209
  H   -3.272169589   1.334861248  -6.407547941
  P   -1.700233712  -1.085729439  -2.180311880
  O   -0.258353740  -0.551786362  -1.639450429
  C    0.824451619  -1.517190960  -1.549878236
  H    1.010079121  -1.997118553  -2.542380153
  H    1.691962855  -0.885549446  -1.272901017
  C    0.400392716  -2.488204232  -0.452724054
  H    0.200698831  -1.978468575   0.521775445
  O   -0.920977382  -2.930632370  -0.921610147
  C   -0.952134959  -4.378400657  -1.122860296
  H   -1.925848420  -4.676477826  -0.656590332
  C    1.259648256  -3.751412436  -0.240237293
  H    2.141410030  -3.806823509  -0.928615459
  C    0.287039374  -4.924336216  -0.417645996
  H    0.783195459  -5.747055483  -0.994895920
  H   -0.033268665  -5.404964640   0.543716888
  O    1.813553741  -3.581131588   1.077279375
  O   -1.716311899  -1.764823106  -3.515314766
  O   -2.763940570  -1.613592627  -1.078788042
  N   -0.992014134  -4.654587849  -2.583520167
  C    0.218417503  -4.627413190  -3.354833534
  N    0.125951545  -5.117830828  -4.675252558
  C   -1.039747740  -5.691755958  -5.255516546
  C   -2.213678630  -5.743285493  -4.382301493
  C   -2.150908093  -5.278958261  -3.106732974
  O    1.264992760  -4.191997824  -2.911465387
  H    1.000329960  -5.043041689  -5.254394098
  O   -0.951828468  -6.050399191  -6.426778167
  C   -3.453148740  -6.357420715  -4.934097187
  H   -4.292386551  -5.647428190  -4.942891228
  H   -3.311081325  -6.704495986  -5.970813843
  H   -3.760433290  -7.240972326  -4.349562591
  H   -3.003389908  -5.366255996  -2.417382553
  P    1.991142596  -4.873117158   2.085245014
  O    3.119636995  -5.719805805   1.236487120
  C    3.340247451  -7.084769299   1.686673869
  H    4.222295686  -7.034349890   2.360377862
  H    2.464690287  -7.491416994   2.237446772
  C    3.637399657  -7.946287773   0.453360186
  H    4.556661211  -8.558456792   0.620148782
  O    2.598681456  -8.960501906   0.333781930
  C    1.741087135  -8.716639411  -0.793152834
  H    1.666355820  -9.702033737  -1.313924897
  C    3.677183316  -7.194283249  -0.901132932
  H    3.816469346  -6.091520715  -0.788476063
  C    2.371369431  -7.586511191  -1.613904572
  H    1.699089139  -6.703849308  -1.722422686
  H    2.556406497  -7.895568326  -2.661923006
  O    4.824724298  -7.563694453  -1.645700662
  O    0.806039986  -5.647366295   2.468512482
  O    2.805683666  -4.108073975   3.262144258
  N    0.381791305  -8.374918650  -0.235776746
  C   -0.652696662  -7.795890779  -1.108043947
  N   -1.854769526  -7.424903350  -0.531708956
  C   -2.105854531  -7.713888409   0.782619706
  C   -1.139932827  -8.375605464   1.615562043
  C    0.088286693  -8.676313604   1.089431352
  O   -0.355413630  -7.611654268  -2.276975041
  N   -3.313595525  -7.292637107   1.286755817
  H   -3.602438344  -7.546433295   2.219546604
  H   -4.027385218  -6.933856878   0.669220321
  H    1.776595558  -2.965837755  -5.745204896
  H    0.877584958  -9.166426199   1.685186286
  H   -1.363694740  -8.605620855   2.655721197
  H    4.845415307  -8.522812204  -1.856131936
  H   -2.330738827  -3.104091502  -4.550701491
  H   -2.417667890  -1.987667372  -0.205895924
  H    3.460010840  -3.398701345   2.992604779
  H    0.349252730  -8.503556362  -8.843959676
  H   -4.119572953   6.087353162  -4.766469269
