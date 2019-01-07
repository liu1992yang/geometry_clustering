%mem=64gb
%nproc=28       
%Chk=snap_129.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_129 

2     1 
  O    1.344035986  -2.254303131  -6.113754160
  C    1.328591780  -3.249551912  -7.128323215
  H    1.450333032  -4.262004208  -6.674279309
  H    2.229556675  -3.026620085  -7.736901763
  C    0.051740370  -3.145005907  -7.976675136
  H   -0.060790940  -2.136067302  -8.436355473
  O    0.214555828  -4.070524105  -9.106031669
  C   -0.901158744  -4.947304818  -9.222413190
  H   -1.111808935  -5.058142462 -10.313213546
  C   -1.252809898  -3.585955381  -7.265079811
  H   -1.043355096  -4.187369244  -6.352879975
  C   -2.010404996  -4.408939786  -8.320402917
  H   -2.720010660  -3.756917398  -8.875437144
  H   -2.645801953  -5.199348232  -7.863150143
  O   -2.008682905  -2.406137172  -6.978646019
  N   -0.436823730  -6.294798658  -8.703584406
  C   -0.961234618  -7.562044881  -9.028176533
  C   -0.401504901  -8.487315475  -8.095484026
  N    0.426788792  -7.775289897  -7.219986991
  C    0.397608458  -6.472837928  -7.584859130
  N   -1.812508270  -7.895813349 -10.045323212
  C   -2.117163728  -9.233555470 -10.123527018
  N   -1.626023432 -10.207434769  -9.217356183
  C   -0.724939241  -9.891304213  -8.120500461
  N   -2.953970569  -9.630959940 -11.133477267
  H   -3.291245632  -8.942467030 -11.804007943
  H   -3.200149825 -10.595423477 -11.300700236
  O   -0.359279323 -10.781423268  -7.390850680
  H    0.967881724  -5.653119835  -7.095646667
  H   -1.905952722 -11.194840187  -9.302498037
  P   -2.304357268  -2.171149718  -5.356476283
  O   -2.833877527  -0.615347080  -5.513550977
  C   -1.831765471   0.406736767  -5.374898582
  H   -2.194865699   1.197549997  -6.065088077
  H   -0.823590534   0.083275515  -5.699487915
  C   -1.859158050   0.868174796  -3.907498230
  H   -1.125364298   0.318725790  -3.260593555
  O   -1.353459922   2.236988491  -3.849253890
  C   -2.396288523   3.163090504  -3.535115798
  H   -1.943784108   3.872179054  -2.794716152
  C   -3.308403096   0.893581889  -3.348725546
  H   -4.054659657   0.435129118  -4.048873253
  C   -3.609619941   2.375901735  -3.055200552
  H   -3.758694498   2.496381853  -1.950937792
  H   -4.571834471   2.688363463  -3.498712283
  O   -3.435685131   0.249369431  -2.073699272
  O   -1.048334552  -2.426026446  -4.596832025
  O   -3.641531707  -3.075327287  -5.261007172
  N   -2.642271400   3.920180593  -4.815038319
  C   -1.693546911   4.131228592  -5.810515156
  C   -2.305981039   4.962156606  -6.803414208
  N   -3.620553688   5.256309502  -6.415845148
  C   -3.814173173   4.644621035  -5.246192855
  N   -0.377408082   3.687931545  -5.941344416
  C    0.312464349   4.129500767  -7.083661983
  N   -0.209771781   4.899381336  -8.025681712
  C   -1.542125920   5.361448083  -7.947713207
  H    1.370477617   3.801662341  -7.192560353
  N   -2.003545378   6.136714841  -8.940684848
  H   -1.416027874   6.396849909  -9.733863604
  H   -2.955646534   6.504167870  -8.921772105
  H   -4.723294771   4.673096644  -4.650546964
  P   -2.953541907  -1.333181046  -1.945944429
  O   -1.445352125  -0.904186395  -1.480706130
  C   -0.423193891  -1.926125727  -1.367004874
  H   -0.307721586  -2.460827390  -2.341352868
  H    0.499096807  -1.339830458  -1.156619873
  C   -0.876642302  -2.806069491  -0.205834092
  H   -1.005797048  -2.241218482   0.748470630
  O   -2.240846863  -3.171212935  -0.628307720
  C   -2.360826079  -4.615351117  -0.789211419
  H   -3.382140538  -4.818337686  -0.380273949
  C   -0.120145890  -4.125640341   0.032151155
  H    0.671487969  -4.291969364  -0.755659591
  C   -1.208326153  -5.214992810   0.012446829
  H   -0.827294587  -6.178450745  -0.389404722
  H   -1.532772383  -5.462787285   1.048693354
  O    0.495622399  -4.008026577   1.323727082
  O   -3.109338230  -2.083906639  -3.230986466
  O   -4.009629288  -1.746821547  -0.793541670
  N   -2.331219391  -4.975882867  -2.235702986
  C   -1.112080218  -5.108840198  -2.973838981
  N   -1.208937358  -5.684391305  -4.257884736
  C   -2.435097539  -6.022039941  -4.916790262
  C   -3.648083674  -5.744922109  -4.129748778
  C   -3.569891331  -5.249617015  -2.870298583
  O   -0.019710251  -4.800179377  -2.525788457
  H   -0.305001523  -5.877550143  -4.744620622
  O   -2.342059526  -6.473038222  -6.042681559
  C   -4.940951497  -6.012064540  -4.815417405
  H   -5.089012410  -5.308800675  -5.655710029
  H   -4.962804082  -7.027078596  -5.248208862
  H   -5.815258093  -5.925647130  -4.156928189
  H   -4.467561990  -5.038050233  -2.271596384
  P    1.712356906  -5.118908503   1.578683229
  O    2.949388559  -4.379846297   0.769694689
  C    3.562360018  -5.142250389  -0.297255147
  H    4.485544424  -5.592669162   0.126623238
  H    2.906675237  -5.950251101  -0.681038924
  C    3.906305029  -4.097873108  -1.371217813
  H    4.649189202  -3.360698295  -0.991578457
  O    4.609562018  -4.814649563  -2.430884200
  C    3.823347587  -4.882363294  -3.625254077
  H    4.554974307  -4.731682590  -4.455614055
  C    2.680437654  -3.422992258  -2.033695670
  H    1.718100118  -3.722581225  -1.540300847
  C    2.715652239  -3.843305926  -3.508176667
  H    1.708685393  -4.218320627  -3.822062557
  H    2.880319738  -2.985632586  -4.193119762
  O    2.670288555  -2.017176606  -1.857545022
  O    1.421297109  -6.492194738   1.161672252
  O    2.055157131  -4.784681262   3.121312969
  N    3.308929141  -6.311147614  -3.726739564
  C    2.218802394  -6.646807718  -4.611367197
  N    1.858784101  -7.970507156  -4.776289680
  C    2.506050120  -8.965251903  -4.081505520
  C    3.575959522  -8.640154066  -3.174150999
  C    3.960139425  -7.332761679  -3.034867322
  O    1.629879641  -5.753979962  -5.228315537
  N    2.093306040 -10.242210522  -4.324659424
  H    2.503010635 -11.034827213  -3.850618598
  H    1.332448517 -10.435693974  -4.968654745
  H    0.601145750  -2.409007858  -5.448415456
  H    4.799870764  -7.030478970  -2.381767750
  H    4.079706081  -9.427371964  -2.611655758
  H    3.440561118  -1.580863831  -2.282424325
  H   -4.048937786  -3.163139202  -4.318422321
  H   -3.672208923  -1.913279269   0.139200827
  H    2.327699010  -3.844360261   3.347336597
  H    0.925363166  -8.178284551  -6.392924527
  H    0.023808148   3.072832617  -5.213950805

