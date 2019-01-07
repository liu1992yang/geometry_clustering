%mem=64gb
%nproc=28       
%Chk=snap_44.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_44 

2     1 
  O    1.996393377  -4.299618506  -6.111587348
  C    2.564679774  -3.385185914  -7.062519328
  H    3.553820119  -3.850349520  -7.280802941
  H    2.733141342  -2.408481718  -6.561511396
  C    1.742258516  -3.236388426  -8.343624772
  H    2.084853353  -2.349837931  -8.934300094
  O    2.046537481  -4.375126134  -9.220276318
  C    0.853603724  -4.999351178  -9.677461820
  H    1.033551514  -5.291806450 -10.740842420
  C    0.200547726  -3.259652590  -8.169554812
  H   -0.124299430  -3.786241064  -7.228254944
  C   -0.304894974  -4.035035149  -9.391857085
  H   -0.497961805  -3.352979515 -10.246267674
  H   -1.271101909  -4.540854777  -9.191346024
  O   -0.344226657  -1.920408240  -8.212503344
  N    0.698436497  -6.270224765  -8.871693898
  C    0.046376461  -7.442007189  -9.296420860
  C    0.035088888  -8.333519289  -8.184145690
  N    0.658913103  -7.683029354  -7.104577905
  C    1.047888794  -6.457066206  -7.518345227
  N   -0.473777134  -7.720341344 -10.534219547
  C   -1.022644979  -8.969462171 -10.655736171
  N   -1.055533581  -9.920074658  -9.602665438
  C   -0.535782442  -9.649283380  -8.273231025
  N   -1.600873932  -9.297054868 -11.857540886
  H   -1.542262152  -8.639041694 -12.632635951
  H   -1.918312665 -10.229976825 -12.074801760
  O   -0.632641243 -10.520756457  -7.437294448
  H    1.597682707  -5.671771803  -6.879932540
  H   -1.475459586 -10.848802483  -9.746653796
  P   -0.200574457  -1.141744432  -6.766108210
  O   -1.639239692  -0.324154650  -6.617629257
  C   -1.516097780   1.103253320  -6.461707629
  H   -2.563069578   1.445740658  -6.597353529
  H   -0.886998737   1.552858187  -7.256690059
  C   -0.969524184   1.419315691  -5.058561378
  H    0.130159792   1.194084285  -4.948217269
  O   -1.029298098   2.865980848  -4.867960605
  C   -1.991815788   3.216926293  -3.873895770
  H   -1.507930104   4.026640115  -3.267727148
  C   -1.814536347   0.778916689  -3.930151005
  H   -2.607442501   0.097991420  -4.320424129
  C   -2.377357143   1.956750428  -3.108842045
  H   -1.899284429   1.959767801  -2.097507441
  H   -3.455103176   1.834892755  -2.906248576
  O   -0.852587520   0.074130159  -3.132192241
  O    0.969719696  -0.269365809  -6.593234528
  O   -0.386769181  -2.385371619  -5.758876541
  N   -3.131794714   3.830522058  -4.642935398
  C   -2.970968285   4.640338428  -5.764092581
  C   -4.271746711   5.118624213  -6.123542726
  N   -5.221644692   4.592810111  -5.236662858
  C   -4.547566670   3.837572711  -4.367441613
  N   -1.833136845   5.000086458  -6.485176318
  C   -2.037528843   5.890132985  -7.554921666
  N   -3.216785452   6.373655843  -7.912459444
  C   -4.400541435   6.022873820  -7.227514938
  H   -1.138520873   6.205824264  -8.132319099
  N   -5.558179369   6.556126303  -7.646573689
  H   -5.591739828   7.207372411  -8.432349456
  H   -6.439228772   6.338192187  -7.179287473
  H   -4.973244066   3.287464703  -3.532802626
  P   -1.308969747  -1.329718655  -2.391648243
  O   -0.629999238  -1.054877355  -0.945537047
  C   -0.186207598  -2.256036660  -0.254438866
  H    0.538725999  -2.827802338  -0.882235161
  H    0.361292538  -1.839458268   0.619007741
  C   -1.442464865  -3.027864800   0.148042557
  H   -1.985737931  -2.559850309   1.001831246
  O   -2.345531627  -2.889130544  -1.004753808
  C   -2.949939624  -4.194391037  -1.367129968
  H   -4.044597206  -3.968096094  -1.346892542
  C   -1.254393288  -4.545819242   0.347293197
  H   -0.309144056  -4.929042373  -0.144867131
  C   -2.497549417  -5.180941518  -0.294830810
  H   -2.269252244  -6.193601242  -0.692693880
  H   -3.284564262  -5.343391384   0.474936372
  O   -1.301804204  -4.800483449   1.763055998
  O   -0.850839247  -2.463459929  -3.249021478
  O   -2.921015406  -1.025229033  -2.302141776
  N   -2.542175298  -4.526256649  -2.746719841
  C   -1.257887732  -5.116336866  -3.027259558
  N   -0.962518838  -5.356023158  -4.384095378
  C   -1.765957188  -4.924969056  -5.479785082
  C   -2.971815155  -4.161126849  -5.112108782
  C   -3.306796900  -3.975635191  -3.813371846
  O   -0.507479402  -5.486352972  -2.143425942
  H   -0.060853793  -5.858117001  -4.583062683
  O   -1.384425389  -5.230778263  -6.597533140
  C   -3.753793833  -3.593269388  -6.242950452
  H   -4.698900604  -3.129573269  -5.935198394
  H   -3.168417350  -2.825121830  -6.777314455
  H   -4.002182652  -4.370316455  -6.985864745
  H   -4.198165493  -3.407943727  -3.520638990
  P   -0.010104385  -5.576630918   2.421756164
  O    0.421929590  -6.447242566   1.100777893
  C    1.687039589  -7.137218334   1.026944226
  H    2.039897431  -7.473280020   2.018556101
  H    1.423808707  -8.030903617   0.417496455
  C    2.713641445  -6.255164718   0.294217045
  H    3.473232364  -5.827623536   0.987637087
  O    3.488484703  -7.144135106  -0.564282632
  C    3.401524234  -6.747660749  -1.946774404
  H    4.438288516  -6.869926885  -2.344157230
  C    2.103095304  -5.179971271  -0.640145330
  H    0.990922851  -5.305917942  -0.758856714
  C    2.843454213  -5.330155619  -1.971051687
  H    2.185084043  -5.109796863  -2.844180079
  H    3.657294394  -4.574458022  -2.043888463
  O    2.347805252  -3.850710323  -0.174032329
  O   -0.182856831  -6.194724510   3.723913411
  O    1.141192691  -4.372825253   2.371922202
  N    2.523327868  -7.789260403  -2.615353130
  C    1.764635318  -7.522384767  -3.820981337
  N    1.085873657  -8.568975587  -4.428573860
  C    1.099075468  -9.831154567  -3.891446252
  C    1.814834042 -10.093762753  -2.668650246
  C    2.519684452  -9.078463459  -2.081500122
  O    1.734082610  -6.400605116  -4.317460264
  N    0.426462789 -10.801025606  -4.573360722
  H    0.396547937 -11.755729474  -4.248516522
  H   -0.033358788 -10.600597054  -5.463905961
  H    1.107227645  -3.996185272  -5.787480850
  H    3.118875446  -9.237700064  -1.166079473
  H    1.811726824 -11.090766447  -2.231305616
  H    2.194083023  -3.791694070   0.803718767
  H   -0.547569099  -2.217579575  -4.705054926
  H   -3.409558683  -1.367523792  -1.488232754
  H    1.477880632  -4.029060499   3.242101277
  H    0.760134317  -8.093742678  -6.135057690
  H   -0.918613620   4.590488301  -6.234557438

