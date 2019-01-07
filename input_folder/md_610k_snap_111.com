%mem=64gb
%nproc=28       
%Chk=snap_111.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_111 

2     1 
  O    3.404382177  -4.156040277  -6.331059464
  C    3.411590636  -2.969373265  -7.118637511
  H    4.006677444  -3.142768621  -8.037502087
  H    3.964560040  -2.252263843  -6.471542953
  C    2.011702959  -2.451137597  -7.462616915
  H    2.006287611  -1.350286216  -7.656030870
  O    1.629599463  -3.017410244  -8.762215833
  C    0.535092414  -3.907550469  -8.633579723
  H   -0.135974722  -3.715110803  -9.507099103
  C    0.906830789  -2.852513271  -6.460920713
  H    1.323790576  -3.366989830  -5.549946462
  C   -0.070730465  -3.741771749  -7.242800336
  H   -1.089972047  -3.294568239  -7.295272374
  H   -0.222628278  -4.716834459  -6.722095200
  O    0.284654503  -1.603685737  -6.084800070
  N    1.107696905  -5.302590642  -8.815237142
  C    0.429407344  -6.513036895  -8.563371567
  C    1.362023070  -7.561776704  -8.816448861
  N    2.583687858  -6.977547796  -9.170109042
  C    2.422870499  -5.623889934  -9.177796277
  N   -0.887817713  -6.680435927  -8.230180007
  C   -1.277951872  -7.982085718  -8.080442605
  N   -0.433779211  -9.082769393  -8.333271719
  C    0.983207537  -8.952507530  -8.651909522
  N   -2.566822512  -8.233223988  -7.611471880
  H   -3.139538753  -7.383151914  -7.425834619
  H   -3.099594651  -8.991879431  -8.028879489
  O    1.642116086  -9.951456108  -8.765230686
  H    3.173285536  -4.875453921  -9.447006233
  H   -0.766190704 -10.045182821  -8.172963615
  P   -0.581167829  -1.686019371  -4.668657869
  O   -0.853173852  -0.049464929  -4.597148738
  C   -0.179135656   0.656052868  -3.523847378
  H    0.515754354   1.363104002  -4.032328255
  H    0.408751941  -0.026664214  -2.871007159
  C   -1.243223952   1.410529564  -2.728916702
  H   -0.961013253   1.485951515  -1.646720696
  O   -1.235114113   2.787818344  -3.216567903
  C   -2.566302221   3.314803374  -3.280790798
  H   -2.577623593   4.210276714  -2.606962866
  C   -2.705043791   0.912772389  -2.888220815
  H   -2.875358288   0.300216557  -3.806801690
  C   -3.544022383   2.197562813  -2.901815758
  H   -3.983057415   2.363284343  -1.887359127
  H   -4.424177965   2.127464368  -3.568064368
  O   -3.070864762   0.180773763  -1.711113395
  O    0.213855622  -2.260196388  -3.562237797
  O   -1.911841784  -2.316316132  -5.353713795
  N   -2.754824064   3.805569915  -4.681081834
  C   -2.448571785   3.195128275  -5.894098268
  C   -2.758889264   4.145574004  -6.924128348
  N   -3.245684644   5.326972004  -6.345182631
  C   -3.233084238   5.129657378  -5.029914141
  N   -1.952388880   1.936874284  -6.220896189
  C   -1.736220094   1.678678511  -7.583642196
  N   -2.003555250   2.526420989  -8.566470751
  C   -2.524940376   3.809571215  -8.294880717
  H   -1.316481655   0.677942687  -7.836780449
  N   -2.760444375   4.632885860  -9.330328039
  H   -2.562071828   4.350349120 -10.289051948
  H   -3.137274932   5.570400338  -9.184500924
  H   -3.541493695   5.840824189  -4.266169744
  P   -2.703302103  -1.439850979  -1.762164585
  O   -1.530475139  -1.411633761  -0.636956675
  C   -0.634405654  -2.563578515  -0.646313873
  H   -0.220151128  -2.723600785  -1.670467882
  H    0.182020206  -2.255780832   0.045116283
  C   -1.450991904  -3.735622490  -0.120848372
  H   -1.744870712  -3.612310321   0.949851349
  O   -2.709350790  -3.620442559  -0.871770339
  C   -3.101306830  -4.905553142  -1.446872680
  H   -4.157215611  -5.026739562  -1.093651095
  C   -0.906619165  -5.153058313  -0.402280798
  H   -0.052730561  -5.169915712  -1.139730572
  C   -2.128781025  -5.944849110  -0.895509113
  H   -1.837543893  -6.735353352  -1.616079189
  H   -2.571998480  -6.500237385  -0.033235638
  O   -0.563837360  -5.770008640   0.854015487
  O   -2.296320362  -1.820388884  -3.159531483
  O   -4.170477283  -1.952573008  -1.291077368
  N   -3.104125642  -4.775434183  -2.927553606
  C   -2.009797455  -5.233839885  -3.735087521
  N   -2.329310491  -5.518762900  -5.083753461
  C   -3.599021452  -5.243274935  -5.694171425
  C   -4.505538046  -4.421837814  -4.870275856
  C   -4.272493147  -4.257330307  -3.546268029
  O   -0.891895893  -5.408029952  -3.290081700
  H   -1.630389935  -6.063004418  -5.630191073
  O   -3.810435922  -5.710139877  -6.794539354
  C   -5.680549638  -3.838294731  -5.569790191
  H   -6.451172881  -3.458255179  -4.884912454
  H   -5.381792382  -3.005463974  -6.226568515
  H   -6.172172634  -4.586657231  -6.219741667
  H   -4.961901957  -3.709560539  -2.883445570
  P    1.004589854  -5.655940991   1.326550353
  O    1.684413195  -6.091880910  -0.110780233
  C    3.120022142  -5.920019754  -0.231063196
  H    3.555959025  -5.358571924   0.617846363
  H    3.539936717  -6.953773734  -0.242824534
  C    3.397010545  -5.206939496  -1.554287234
  H    4.366835602  -4.652052202  -1.508451954
  O    3.617520240  -6.261211337  -2.555841570
  C    2.952259094  -5.949419213  -3.782789943
  H    3.695297615  -6.150603534  -4.589915782
  C    2.276135603  -4.317548337  -2.142264848
  H    1.257196444  -4.608842837  -1.780620033
  C    2.430691310  -4.519142460  -3.657916855
  H    1.486109225  -4.331296790  -4.209180555
  H    3.170485566  -3.803004896  -4.084903022
  O    2.539297075  -2.972847062  -1.752690798
  O    1.399921312  -6.317819052   2.557568932
  O    1.203734455  -4.012314389   1.283128788
  N    1.797687846  -6.918445340  -3.950378666
  C    1.086861520  -6.918752752  -5.220681437
  N   -0.001640827  -7.752664095  -5.377573134
  C   -0.432688947  -8.516224220  -4.321625809
  C    0.225621251  -8.491884283  -3.051784383
  C    1.322101677  -7.675317410  -2.892564320
  O    1.476838000  -6.164266658  -6.111598975
  N   -1.584344796  -9.239980812  -4.540946928
  H   -1.865674730  -9.971419580  -3.905341985
  H   -1.961564771  -9.280335421  -5.478926744
  H    2.786836310  -4.845998684  -6.680389315
  H    1.859884036  -7.592575215  -1.929597222
  H   -0.125459050  -9.105788789  -2.225490974
  H    1.933086898  -2.373024080  -2.263987301
  H   -2.653481523  -2.611771663  -4.699877742
  H   -4.263698361  -2.306426207  -0.350308827
  H    1.626342450  -3.570190899   2.065549045
  H    3.426491335  -7.503411304  -9.389833614
  H   -1.663534567   1.254282599  -5.488386413
