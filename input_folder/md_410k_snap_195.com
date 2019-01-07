%mem=64gb
%nproc=28       
%Chk=snap_195.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_195 

2     1 
  O    0.776724671  -1.859030029  -6.470701689
  C    0.810083624  -2.431247731  -7.778559678
  H    1.400771739  -3.374123069  -7.780439710
  H    1.342037062  -1.684104179  -8.400827509
  C   -0.626823802  -2.660679925  -8.274541356
  H   -1.034786414  -1.766663434  -8.804794651
  O   -0.560430615  -3.699431941  -9.306026924
  C   -1.477207055  -4.754021097  -9.039341856
  H   -1.932373665  -5.041339463 -10.019073960
  C   -1.638779088  -3.163932343  -7.213631015
  H   -1.153896070  -3.491729251  -6.261664623
  C   -2.414796634  -4.287182719  -7.922610234
  H   -3.367252075  -3.876717301  -8.338430633
  H   -2.741108249  -5.092944963  -7.241755682
  O   -2.556078251  -2.072063596  -7.023380458
  N   -0.648778758  -5.917603040  -8.547023162
  C   -1.010588713  -7.282162506  -8.557676487
  C   -0.059174602  -7.955277683  -7.731617216
  N    0.858854889  -7.010765144  -7.258530469
  C    0.489689597  -5.798291314  -7.723935095
  N   -2.060379921  -7.877865448  -9.187774718
  C   -2.144287309  -9.242453839  -8.993794487
  N   -1.211592503  -9.990886375  -8.229643881
  C   -0.152772376  -9.368522031  -7.467338363
  N   -3.163013356  -9.902351784  -9.624506919
  H   -3.836436494  -9.375255221 -10.178088634
  H   -3.365613996 -10.878836574  -9.465474471
  O    0.508611028 -10.035799303  -6.700644470
  H    1.010057174  -4.844721862  -7.510484197
  H   -1.380250508 -10.989274601  -8.035815281
  P   -2.794927005  -1.568705711  -5.475391360
  O   -2.635808054   0.067226081  -5.647203988
  C   -1.282597721   0.535932089  -5.807572093
  H   -1.436510953   1.544941799  -6.243289888
  H   -0.709508487  -0.086991931  -6.531443973
  C   -0.559706334   0.612632723  -4.452080568
  H    0.151282358  -0.238627739  -4.302374268
  O    0.327100502   1.777333281  -4.497854879
  C   -0.089142993   2.778878007  -3.570798524
  H    0.857457284   3.215369345  -3.160600194
  C   -1.497553196   0.824074688  -3.235196242
  H   -2.581310144   0.871370346  -3.518378252
  C   -1.014866031   2.118040257  -2.555969708
  H   -0.458906655   1.862703865  -1.617733478
  H   -1.851132495   2.744411629  -2.199687626
  O   -1.242662397  -0.220257457  -2.276338102
  O   -1.758239833  -2.173902605  -4.549993277
  O   -4.378817789  -1.838952796  -5.396647480
  N   -0.773088710   3.831230013  -4.401897322
  C   -0.285466197   4.317876314  -5.612380523
  C   -1.233388990   5.278595069  -6.089641427
  N   -2.303358032   5.363716694  -5.188026848
  C   -2.033806812   4.506530201  -4.203029738
  N    0.877572573   4.027297236  -6.324462076
  C    1.071369873   4.755642271  -7.513872867
  N    0.228887534   5.658677315  -7.986947224
  C   -0.973809513   5.972757922  -7.315588060
  H    2.005578792   4.551835266  -8.084581170
  N   -1.789401663   6.886454964  -7.862050862
  H   -1.554484446   7.358824737  -8.736087852
  H   -2.666590225   7.148382210  -7.408991344
  H   -2.653091136   4.320140971  -3.329668453
  P   -2.224376404  -1.552415620  -2.387466176
  O   -1.067718300  -2.683463074  -2.339379897
  C   -1.485626113  -4.061462446  -2.535317461
  H   -2.360913281  -4.158682418  -3.205789785
  H   -0.602124615  -4.494212221  -3.059452670
  C   -1.749833530  -4.674453766  -1.156116621
  H   -2.016028190  -3.904437553  -0.384034926
  O   -2.951877428  -5.486189048  -1.292481007
  C   -2.663630631  -6.873650603  -1.103273132
  H   -3.535503016  -7.265785159  -0.525268869
  C   -0.630450423  -5.612240934  -0.638822246
  H    0.234910295  -5.697268887  -1.340117632
  C   -1.313399384  -6.968787821  -0.389004306
  H   -0.686766102  -7.816342765  -0.726699298
  H   -1.455014124  -7.128021077   0.703562136
  O   -0.242809149  -5.065314107   0.641920355
  O   -3.418103070  -1.288112989  -3.272431424
  O   -2.743013078  -1.734412293  -0.847240569
  N   -2.667460564  -7.529517114  -2.464386494
  C   -1.574474209  -7.371613777  -3.360477084
  N   -1.697067350  -7.983717199  -4.631071015
  C   -2.851910810  -8.709477253  -5.084248108
  C   -3.938347191  -8.800358110  -4.102484450
  C   -3.829759229  -8.218938303  -2.880083617
  O   -0.561436405  -6.749196511  -3.082551716
  H   -0.856909331  -7.947116104  -5.221020470
  O   -2.811879414  -9.150522196  -6.217168007
  C   -5.147677802  -9.574556605  -4.499106593
  H   -5.102342118  -9.906332715  -5.550192179
  H   -5.263820567 -10.485622466  -3.888634023
  H   -6.072483294  -8.986623059  -4.393719670
  H   -4.647420397  -8.269922653  -2.145253668
  P    1.374336186  -4.955277020   0.917620243
  O    1.585656795  -3.360697366   0.572717651
  C    2.948636483  -2.938574402   0.313525491
  H    3.126662700  -2.161165245   1.090095357
  H    3.693870037  -3.754833163   0.428001364
  C    3.031715422  -2.331552720  -1.088480674
  H    3.738450168  -1.461713590  -1.097031330
  O    3.668505216  -3.314252010  -1.966064288
  C    2.968530945  -3.441860469  -3.208878535
  H    3.726934140  -3.233812851  -4.004231209
  C    1.691351173  -1.945079722  -1.746497646
  H    0.800078059  -2.314295111  -1.186695846
  C    1.775601814  -2.487492737  -3.185306625
  H    0.820827289  -2.958628034  -3.497735490
  H    1.942408133  -1.648227536  -3.892473189
  O    1.681905706  -0.511599523  -1.771961330
  O    2.266933585  -5.874718206   0.200405237
  O    1.383251003  -5.017393894   2.538707388
  N    2.571207497  -4.897667364  -3.341776691
  C    1.963825964  -5.340279454  -4.578880584
  N    1.867159768  -6.700578873  -4.819656289
  C    2.192360656  -7.613878120  -3.840255575
  C    2.740871268  -7.176083043  -2.592608608
  C    2.959854515  -5.838304842  -2.385804834
  O    1.565217985  -4.502849724  -5.391076938
  N    1.956868501  -8.929571379  -4.134649889
  H    2.254349539  -9.667968640  -3.512233812
  H    1.619239985  -9.218005113  -5.051026044
  H    0.862936450  -2.585317859  -5.786828129
  H    3.450093644  -5.454785218  -1.465408041
  H    2.982656524  -7.884708802  -1.797679556
  H    0.751548688  -0.198097683  -1.864151598
  H   -4.822542935  -1.641762409  -4.487677372
  H   -3.621755004  -1.352205406  -0.585039820
  H    0.685470966  -4.502198722   3.035604317
  H    1.558481194  -7.171805762  -6.466276425
  H    1.501853729   3.274701570  -6.005868563
