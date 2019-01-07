%mem=64gb
%nproc=28       
%Chk=snap_103.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_103 

2     1 
  O    3.519989322  -4.781311016  -6.808467459
  C    3.621954574  -3.372566693  -6.615951580
  H    4.521440872  -3.108410765  -7.214909742
  H    3.832628000  -3.159722771  -5.546312534
  C    2.393507455  -2.592919142  -7.100060415
  H    2.602957111  -1.491969781  -7.139289058
  O    2.136300638  -2.927982907  -8.501380947
  C    0.910932839  -3.629416108  -8.640742543
  H    0.475996547  -3.299708441  -9.616168838
  C    1.075108322  -2.866305904  -6.335581987
  H    1.194748878  -3.581132706  -5.477963677
  C    0.078797988  -3.381482035  -7.385116547
  H   -0.728508124  -2.644430445  -7.575642085
  H   -0.457374905  -4.287894711  -7.023088934
  O    0.674190518  -1.558188983  -5.870825917
  N    1.285109718  -5.092803773  -8.792080556
  C    0.459969975  -6.219276870  -8.587133662
  C    1.286609158  -7.366847069  -8.749238937
  N    2.597910502  -6.923943566  -8.996363875
  C    2.592889059  -5.561309691  -9.001430334
  N   -0.888110078  -6.245958205  -8.342549374
  C   -1.422917350  -7.494245163  -8.205846251
  N   -0.693007005  -8.681394593  -8.410865698
  C    0.749644232  -8.704924484  -8.629189607
  N   -2.748850815  -7.601962629  -7.780414539
  H   -3.220374550  -6.692052477  -7.606778921
  H   -3.350540532  -8.281367150  -8.237064167
  O    1.296787256  -9.774310191  -8.699672871
  H    3.457186578  -4.894102311  -9.130108356
  H   -1.142502992  -9.600418956  -8.299253705
  P   -0.243979796  -1.527139464  -4.497349280
  O   -0.465968807   0.111352655  -4.515232269
  C    0.209919745   0.925424219  -3.529852607
  H    0.871183441   1.594175671  -4.131465572
  H    0.822764575   0.313975336  -2.835802816
  C   -0.841237915   1.744205599  -2.781287594
  H   -0.489185372   1.996878573  -1.748343449
  O   -0.950730720   3.029148963  -3.474976677
  C   -2.308751354   3.486766952  -3.490506747
  H   -2.297261238   4.503822521  -3.021571922
  C   -2.286673448   1.180003724  -2.752191361
  H   -2.505082623   0.518761740  -3.628510516
  C   -3.167255741   2.437596612  -2.774534775
  H   -3.408041936   2.747578115  -1.728670104
  H   -4.155524181   2.265425827  -3.237567394
  O   -2.499446542   0.505527735  -1.510509483
  O    0.447608345  -2.154908659  -3.355253281
  O   -1.630868871  -1.990257422  -5.221227539
  N   -2.694965321   3.660769169  -4.926258056
  C   -2.653549991   2.755942103  -5.983732508
  C   -3.151496784   3.443951285  -7.140113871
  N   -3.477524510   4.764514502  -6.796754493
  C   -3.202039353   4.891818160  -5.501396857
  N   -2.270102859   1.421660943  -6.068553455
  C   -2.359535980   0.817099673  -7.331190006
  N   -2.823245860   1.409048324  -8.423637196
  C   -3.237087242   2.757539009  -8.393031598
  H   -2.031774538  -0.245821703  -7.394098453
  N   -3.691170804   3.312500113  -9.529643950
  H   -3.732591939   2.786094539 -10.400984202
  H   -3.998442859   4.285913619  -9.551524935
  H   -3.328781406   5.790352083  -4.900837613
  P   -2.355102225  -1.150410689  -1.601555247
  O   -1.249143485  -1.312225948  -0.427456548
  C   -0.477694953  -2.550161050  -0.466558072
  H   -0.022150183  -2.692814852  -1.474693842
  H    0.326078346  -2.366127849   0.282532209
  C   -1.437458228  -3.655381598  -0.049712775
  H   -1.767453873  -3.561207364   1.013617650
  O   -2.641108603  -3.364168315  -0.843371121
  C   -3.151070201  -4.573523027  -1.485984572
  H   -4.226821303  -4.588295574  -1.175058996
  C   -1.034432978  -5.105889567  -0.393203505
  H   -0.171577159  -5.173558402  -1.117886911
  C   -2.321681229  -5.737029794  -0.949243656
  H   -2.097916153  -6.528276153  -1.692399188
  H   -2.851598072  -6.271192549  -0.123700642
  O   -0.784310689  -5.817382690   0.833805257
  O   -1.985512816  -1.523300731  -3.012763845
  O   -3.887716874  -1.500631888  -1.190141848
  N   -3.080284810  -4.387561323  -2.959698424
  C   -1.999224820  -4.915205337  -3.744372287
  N   -2.263249578  -5.059415489  -5.124858323
  C   -3.450002377  -4.595093970  -5.784413986
  C   -4.340094805  -3.780485010  -4.938170954
  C   -4.166356568  -3.734001263  -3.594797870
  O   -0.937732584  -5.252452791  -3.254230829
  H   -1.581009429  -5.626369522  -5.673348861
  O   -3.614099541  -4.911594318  -6.944364655
  C   -5.452904536  -3.069841617  -5.624174102
  H   -6.437854957  -3.452616631  -5.307785660
  H   -5.439810832  -1.989019941  -5.429905881
  H   -5.414611202  -3.205124095  -6.720488162
  H   -4.851253437  -3.179501711  -2.933101581
  P    0.779061033  -5.907174589   1.326576176
  O    1.439746908  -6.331033264  -0.124939429
  C    2.885043947  -6.302807062  -0.213346582
  H    3.362191805  -5.866910661   0.685558761
  H    3.193829478  -7.370591232  -0.304205347
  C    3.269728891  -5.519572935  -1.468372689
  H    4.265358161  -5.027334651  -1.338440250
  O    3.485900016  -6.519031610  -2.527319254
  C    2.879324758  -6.109433439  -3.753067576
  H    3.633729024  -6.330224912  -4.545224297
  C    2.235292804  -4.528776212  -2.051641755
  H    1.182420768  -4.774423159  -1.758030040
  C    2.452105509  -4.652980871  -3.571549437
  H    1.538346452  -4.374875639  -4.142438517
  H    3.251533387  -3.957190373  -3.902890604
  O    2.573196802  -3.231579224  -1.578898162
  O    1.080943736  -6.679739724   2.518651501
  O    1.157540297  -4.296479276   1.375373720
  N    1.670416287  -6.991326071  -3.999310474
  C    0.841322702  -6.738688950  -5.165905869
  N   -0.258829302  -7.530429040  -5.402256809
  C   -0.633989408  -8.465920059  -4.468950373
  C    0.123692039  -8.676201584  -3.270600457
  C    1.258070659  -7.929514142  -3.061710692
  O    1.150190720  -5.805733684  -5.912927016
  N   -1.803469248  -9.134959590  -4.733888919
  H   -2.089901315  -9.930689424  -4.183062582
  H   -2.255405049  -9.001876131  -5.629226151
  H    2.729473243  -5.165651130  -6.323815378
  H    1.880915470  -8.042945410  -2.156096328
  H   -0.188296447  -9.416671669  -2.535000195
  H    2.000534825  -2.557913310  -2.040524290
  H   -2.394838487  -2.245468548  -4.579692214
  H   -4.051113541  -1.915223342  -0.285431612
  H    1.620849566  -3.945082286   2.179430834
  H    3.392712897  -7.544257520  -9.124104072
  H   -1.793421542   0.934876805  -5.269529698
