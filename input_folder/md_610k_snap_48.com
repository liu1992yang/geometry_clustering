%mem=64gb
%nproc=28       
%Chk=snap_48.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_48 

2     1 
  O    2.170015862  -4.277066534  -5.987504899
  C    2.725394062  -3.301684036  -6.879385774
  H    3.740942184  -3.712233918  -7.083976310
  H    2.829651551  -2.336104703  -6.340925122
  C    1.936613703  -3.142310380  -8.180065105
  H    2.256854896  -2.223389889  -8.731599263
  O    2.314019547  -4.243884019  -9.077361975
  C    1.163184926  -4.888694812  -9.602252361
  H    1.392190952  -5.132871688 -10.668467311
  C    0.393692189  -3.238877408  -8.052616831
  H    0.077576667  -3.815670782  -7.138992129
  C   -0.041482857  -3.981014480  -9.322015609
  H   -0.230212450  -3.270639328 -10.154891663
  H   -0.993804921  -4.530134623  -9.179410505
  O   -0.220376984  -1.933449338  -8.062203362
  N    1.008898068  -6.197488694  -8.855455139
  C    0.273020244  -7.303096861  -9.319907029
  C    0.213015266  -8.240794800  -8.248137200
  N    0.893218578  -7.681735327  -7.151434413
  C    1.351513280  -6.461134860  -7.512696286
  N   -0.284632474  -7.480401010 -10.559382885
  C   -0.942088975  -8.671142371 -10.725686959
  N   -1.039226312  -9.661085326  -9.714258028
  C   -0.471678634  -9.497813365  -8.386308513
  N   -1.573872234  -8.885191580 -11.925898274
  H   -1.467407919  -8.196755462 -12.669782887
  H   -1.969436492  -9.777351044 -12.185220212
  O   -0.638720542 -10.398095711  -7.592748193
  H    1.930776660  -5.737614934  -6.853492969
  H   -1.546238228 -10.541876886  -9.887016434
  P   -0.162451896  -1.189890069  -6.593203638
  O   -1.647143276  -0.447126643  -6.514543504
  C   -1.584710021   0.988764960  -6.397128645
  H   -2.628902306   1.292379188  -6.617399902
  H   -0.914001356   1.440074887  -7.155690563
  C   -1.152410749   1.343170080  -4.964000454
  H   -0.063830395   1.136922127  -4.757052287
  O   -1.241143243   2.788713760  -4.798728275
  C   -2.334245734   3.144493508  -3.951220326
  H   -1.939805838   3.960003406  -3.290035996
  C   -2.085342144   0.705823909  -3.905193730
  H   -2.783777501  -0.047339188  -4.341150802
  C   -2.820664300   1.886568176  -3.237496078
  H   -2.547170855   1.924066982  -2.154003685
  H   -3.914090977   1.737359946  -3.246863789
  O   -1.186441029   0.118618690  -2.953630378
  O    0.953782103  -0.259648295  -6.368649386
  O   -0.312801913  -2.446864848  -5.599683417
  N   -3.348881239   3.754111494  -4.883993319
  C   -3.018216841   4.502109522  -6.011799618
  C   -4.243964606   5.004748920  -6.554391327
  N   -5.316385216   4.560108664  -5.768716820
  C   -4.786193348   3.828569426  -4.786767582
  N   -1.787110133   4.785256045  -6.601994640
  C   -1.822433333   5.621041672  -7.732270888
  N   -2.928457554   6.120956317  -8.261194519
  C   -4.200072415   5.849230389  -7.711243399
  H   -0.846589437   5.872986913  -8.206792096
  N   -5.276212776   6.394252016  -8.298552203
  H   -5.188139312   7.001018147  -9.114873884
  H   -6.215656036   6.230465027  -7.934132620
  H   -5.335427931   3.341429134  -3.986111762
  P   -1.556308833  -1.348959672  -2.293222402
  O   -1.065911521  -1.032700928  -0.784178928
  C   -0.555223803  -2.184819766  -0.051437263
  H    0.310954793  -2.629979315  -0.580893496
  H   -0.186036068  -1.723227861   0.894024500
  C   -1.729398806  -3.132486080   0.184598003
  H   -2.394920470  -2.797419816   1.013560215
  O   -2.551761451  -3.033142056  -1.034317210
  C   -3.010872881  -4.373551371  -1.460654462
  H   -4.120262192  -4.247226044  -1.524032800
  C   -1.367507661  -4.633193613   0.300685864
  H   -0.392546900  -4.865220031  -0.209126840
  C   -2.550760430  -5.344245122  -0.379658449
  H   -2.272325076  -6.344623435  -0.766746541
  H   -3.348753692  -5.544395919   0.372063944
  O   -1.379848610  -5.042567934   1.665647568
  O   -0.901147753  -2.423926224  -3.098643696
  O   -3.194025496  -1.228038826  -2.382390084
  N   -2.471530666  -4.643721641  -2.812133846
  C   -1.167896320  -5.215310628  -3.015512029
  N   -0.789904854  -5.430442707  -4.355803782
  C   -1.532660481  -4.990758215  -5.491488709
  C   -2.731565423  -4.194028460  -5.188044175
  C   -3.153820237  -4.048739705  -3.909986460
  O   -0.457330013  -5.595619927  -2.098418320
  H    0.110501314  -5.946413875  -4.515876048
  O   -1.102573669  -5.310381797  -6.587981512
  C   -3.416266042  -3.543373374  -6.338184775
  H   -4.370878062  -4.036555114  -6.577234353
  H   -3.617264096  -2.478232323  -6.150041412
  H   -2.802071329  -3.585140895  -7.254546954
  H   -4.055966629  -3.474254262  -3.660945936
  P    0.068459531  -5.021421159   2.479030954
  O    0.822032147  -6.437390944   2.140656803
  C    1.045836579  -6.960085030   0.824282376
  H    1.110081457  -8.057617500   1.019277036
  H    0.207125308  -6.778518451   0.131119868
  C    2.398819826  -6.431706059   0.329894547
  H    3.068882553  -6.222554100   1.200669519
  O    3.080207441  -7.510421893  -0.365542401
  C    3.273940391  -7.201352993  -1.753730332
  H    4.306792190  -7.565309218  -1.983849109
  C    2.317752258  -5.242560574  -0.660942217
  H    1.267255549  -4.913018494  -0.882548338
  C    3.061773246  -5.698422413  -1.918287681
  H    2.498602391  -5.425140994  -2.847127727
  H    4.027250652  -5.170610960  -2.049360026
  O    2.880830418  -4.077860425  -0.053451885
  O   -0.022050754  -4.764426922   3.901614805
  O    0.782925213  -3.900544877   1.501380531
  N    2.300467391  -8.089925341  -2.500144666
  C    1.723361227  -7.739588412  -3.779295655
  N    0.996927753  -8.694800417  -4.471918049
  C    0.777675648  -9.942284307  -3.943368317
  C    1.288558356 -10.275798673  -2.636570291
  C    2.044504184  -9.356142182  -1.963432848
  O    1.875380329  -6.619955901  -4.260990376
  N    0.067338178 -10.816894065  -4.707928340
  H   -0.108233452 -11.766255037  -4.413893961
  H   -0.238929969 -10.561691146  -5.651498418
  H    1.254543989  -4.033714827  -5.694858175
  H    2.485077931  -9.571418604  -0.971835368
  H    1.088247454 -11.254984173  -2.202272467
  H    3.858268358  -4.146000315   0.056060936
  H   -0.551245655  -2.280271344  -4.552709277
  H   -3.720947961  -1.547485727  -1.584489524
  H    1.779573373  -3.849714826   1.377079746
  H    0.951656202  -8.131744210  -6.199091563
  H   -0.924955378   4.372370358  -6.207447550
