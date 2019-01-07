%mem=64gb
%nproc=28       
%Chk=snap_40.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_40 

2     1 
  O    1.485659258  -4.018025541  -6.501672912
  C    1.783012564  -3.520128033  -7.801598220
  H    2.098800281  -4.422377362  -8.365006667
  H    2.642214394  -2.825504609  -7.740967412
  C    0.562939514  -2.856601200  -8.462138946
  H    0.697037512  -1.761010605  -8.610987484
  O    0.488115028  -3.381508514  -9.826619242
  C   -0.706531365  -4.120494088 -10.022714451
  H   -0.993735313  -3.956904478 -11.092763494
  C   -0.804909302  -3.175019928  -7.801122382
  H   -0.713570013  -3.939353638  -6.996822054
  C   -1.696270959  -3.687281727  -8.947522893
  H   -2.345782458  -2.854412748  -9.308647046
  H   -2.411605602  -4.465136190  -8.620026077
  O   -1.356321262  -1.941383097  -7.328871025
  N   -0.327580298  -5.584360589  -9.891327776
  C    0.263746718  -6.336195064 -10.934762532
  C    0.666792163  -7.580822068 -10.371528725
  N    0.326746402  -7.562818036  -9.007668157
  C   -0.252829096  -6.366611842  -8.726973034
  N    0.403794042  -5.971354286 -12.239886551
  C    1.010971457  -6.915622071 -13.039641576
  N    1.461562082  -8.164506476 -12.558339357
  C    1.317170767  -8.595021706 -11.168960509
  N    1.172493106  -6.599980830 -14.361096243
  H    0.854679180  -5.693765127 -14.702117136
  H    1.609117787  -7.217643069 -15.029482955
  O    1.735313594  -9.680987286 -10.866226188
  H   -0.655044567  -6.060529056  -7.714413445
  H    1.922069944  -8.837937495 -13.189140811
  P   -1.361014601  -1.790574272  -5.665613108
  O   -1.933520644  -0.238357527  -5.674527776
  C   -0.924916866   0.766120085  -5.410219002
  H   -0.988430293   1.461318927  -6.276230177
  H    0.106326140   0.350560576  -5.361269689
  C   -1.298438345   1.478323091  -4.105138707
  H   -0.446988381   1.509340402  -3.379915779
  O   -1.503103214   2.892213062  -4.432874421
  C   -2.816149084   3.311056235  -4.097625413
  H   -2.702947328   4.327467616  -3.656999875
  C   -2.611827097   0.980043128  -3.439212363
  H   -3.173231845   0.234025961  -4.056666504
  C   -3.435072673   2.253986843  -3.184578940
  H   -3.335680403   2.537596231  -2.107383656
  H   -4.519241598   2.096467743  -3.329738335
  O   -2.328983793   0.439125315  -2.146526180
  O    0.002741184  -2.015293690  -5.122333626
  O   -2.615700161  -2.786709782  -5.412292328
  N   -3.576036236   3.423742312  -5.393383545
  C   -3.993837542   4.587409678  -6.031624496
  C   -4.648573238   4.182668818  -7.241671640
  N   -4.595242275   2.786015117  -7.360173467
  C   -3.952959755   2.344530592  -6.279448446
  N   -3.886682965   5.938306840  -5.694881222
  C   -4.442750233   6.856875860  -6.609660053
  N   -5.067676218   6.521823963  -7.723812443
  C   -5.219058643   5.170501380  -8.104681924
  H   -4.342360021   7.938212355  -6.360511401
  N   -5.884474566   4.903714423  -9.239461865
  H   -6.272597123   5.650068118  -9.817717444
  H   -6.018161432   3.943135717  -9.556071950
  H   -3.701587191   1.303148387  -6.056407686
  P   -1.736823835  -1.117170055  -2.131960372
  O   -0.222726448  -0.637227552  -1.761193957
  C    0.832278194  -1.635170503  -1.773236504
  H    0.888094572  -2.137914938  -2.770921313
  H    1.744740860  -1.027933828  -1.608821115
  C    0.501495096  -2.569985788  -0.613243278
  H    0.445327091  -2.041206353   0.370212664
  O   -0.888784232  -2.947229144  -0.911900557
  C   -1.014518518  -4.388722083  -1.111259838
  H   -1.941184365  -4.644535472  -0.537126487
  C    1.314094389  -3.874971191  -0.488200424
  H    2.115811414  -3.971037312  -1.264217844
  C    0.270430809  -4.996462791  -0.557888888
  H    0.659123913  -5.833880528  -1.192461245
  H    0.041871756  -5.468978853   0.434477746
  O    2.006875491  -3.744133720   0.766781941
  O   -1.934468267  -1.797960557  -3.451556371
  O   -2.711285367  -1.602583529  -0.932466321
  N   -1.235967487  -4.654566065  -2.558361747
  C   -0.120291627  -4.694046003  -3.456566511
  N   -0.380349334  -5.192488668  -4.753576778
  C   -1.637581900  -5.688491892  -5.200552626
  C   -2.710842698  -5.665462763  -4.205831308
  C   -2.482982549  -5.202721194  -2.948583157
  O    0.993522510  -4.312710794  -3.144452967
  H    0.433892731  -5.196112398  -5.408834261
  O   -1.701364447  -6.052099030  -6.372428288
  C   -4.033132743  -6.213894670  -4.615237180
  H   -4.855897870  -5.512678593  -4.418763524
  H   -4.063921649  -6.448139680  -5.692769712
  H   -4.255522409  -7.154064972  -4.081557887
  H   -3.259724954  -5.231262789  -2.169448198
  P    2.268137714  -5.062960391   1.720794822
  O    3.334150240  -5.880145174   0.771135734
  C    3.640573325  -7.238011639   1.193334092
  H    4.636057981  -7.164405885   1.683518352
  H    2.893820189  -7.640777447   1.910567885
  C    3.704272189  -8.120117400  -0.059226081
  H    4.594495291  -8.794045231  -0.012395387
  O    2.590710860  -9.058515518  -0.033246284
  C    1.630005568  -8.779059055  -1.064785476
  H    1.465698388  -9.757527011  -1.578646055
  C    3.618715480  -7.363600882  -1.409500192
  H    3.840961033  -6.272870759  -1.314220644
  C    2.210061299  -7.667149382  -1.946351400
  H    1.577280045  -6.749369328  -1.946764746
  H    2.233093804  -7.961915741  -3.013546160
  O    4.637133823  -7.795691152  -2.295627248
  O    1.115821134  -5.844753201   2.180820029
  O    3.175789286  -4.333936809   2.851290302
  N    0.345284527  -8.392474530  -0.377472516
  C   -0.762812440  -7.828973674  -1.166781835
  N   -1.904018825  -7.432935768  -0.493427358
  C   -2.030065433  -7.672238361   0.848101388
  C   -0.989872182  -8.303045513   1.610156536
  C    0.181913671  -8.634296199   0.981774391
  O   -0.569629844  -7.672202170  -2.361675946
  N   -3.181151366  -7.216035519   1.450072183
  H   -3.386725095  -7.452518331   2.409408041
  H   -3.959686522  -6.904458978   0.886907452
  H    1.332652526  -3.266133895  -5.841487607
  H    1.023791913  -9.102291291   1.518152689
  H   -1.110526943  -8.480530555   2.677289156
  H    4.568974311  -8.751027600  -2.512197899
  H   -2.794530229  -3.037609895  -4.431200778
  H   -2.303146826  -1.925278839  -0.068144775
  H    3.845522266  -3.654314212   2.549636021
  H    0.482602831  -8.342062764  -8.374087076
  H   -3.393068235   6.245301064  -4.853565543
