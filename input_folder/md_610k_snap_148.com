%mem=64gb
%nproc=28       
%Chk=snap_148.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_148 

2     1 
  O    0.694275748  -5.400184806  -4.907495367
  C    1.031106503  -4.055739929  -4.541273479
  H    2.096621407  -4.048062111  -4.237203433
  H    0.398598339  -3.712362493  -3.696796740
  C    0.809303804  -3.160766303  -5.766292602
  H    0.993180364  -2.091536958  -5.512046369
  O    1.857287073  -3.401377460  -6.761252096
  C    1.331645657  -4.015795961  -7.924768956
  H    1.868056999  -3.556511107  -8.791936438
  C   -0.546610671  -3.415012098  -6.478933911
  H   -1.180968876  -4.160264993  -5.943762476
  C   -0.190901359  -3.857843907  -7.905582634
  H   -0.526225020  -3.080919496  -8.631630869
  H   -0.730167298  -4.778899849  -8.203576533
  O   -1.288420836  -2.173137433  -6.652854794
  N    1.763273541  -5.464256971  -7.848438466
  C    1.613585823  -6.449423531  -8.839127491
  C    2.208006523  -7.640003574  -8.325983858
  N    2.693486868  -7.362169961  -7.037973519
  C    2.423622019  -6.067719502  -6.755866258
  N    1.032089458  -6.326599360 -10.074868380
  C    1.063862776  -7.466549950 -10.837083558
  N    1.646360858  -8.685712549 -10.403960756
  C    2.254127294  -8.854129745  -9.095840886
  N    0.478148097  -7.418762955 -12.076469228
  H    0.076399704  -6.543428944 -12.406181431
  H    0.524806060  -8.180852865 -12.737159918
  O    2.711635421  -9.940026229  -8.815354521
  H    2.662863684  -5.551070414  -5.812287742
  H    1.651918099  -9.514220026 -11.016681960
  P   -1.890600665  -1.547212303  -5.247149759
  O   -1.678870138   0.083111504  -5.530343710
  C   -0.576480190   0.720027894  -4.849426022
  H    0.003902896   1.207869578  -5.669118728
  H    0.069368452  -0.013919362  -4.320484354
  C   -1.104148003   1.778394506  -3.876081353
  H   -0.435010569   1.863707733  -2.979827571
  O   -0.972708312   3.063988758  -4.557925165
  C   -2.118691403   3.892737510  -4.334735245
  H   -1.720782811   4.861565744  -3.936858851
  C   -2.589808530   1.685219575  -3.442569362
  H   -3.214379122   1.066780814  -4.138649116
  C   -3.070427695   3.140286769  -3.399551033
  H   -3.005261858   3.524664025  -2.351248576
  H   -4.140882665   3.251159551  -3.657965318
  O   -2.641395183   1.187312080  -2.086324236
  O   -1.107799750  -2.110141464  -4.121439441
  O   -3.465048500  -1.617030775  -5.627089651
  N   -2.717973975   4.167191289  -5.675775811
  C   -3.209641709   3.285638550  -6.636760009
  C   -3.632810788   4.085760923  -7.749704306
  N   -3.382958588   5.440251331  -7.481458483
  C   -2.838580296   5.485331442  -6.269091096
  N   -3.366693841   1.903278399  -6.660301147
  C   -3.930291551   1.351989467  -7.823888001
  N   -4.339971790   2.053568755  -8.870906470
  C   -4.206191352   3.456973392  -8.899989205
  H   -4.048636862   0.242262717  -7.838144216
  N   -4.618231395   4.116622854  -9.996257375
  H   -5.028326704   3.626558320 -10.789771175
  H   -4.540739781   5.132356307 -10.057600176
  H   -2.513438734   6.378947668  -5.739997188
  P   -2.885031398  -0.435399172  -2.013272930
  O   -1.598823640  -1.005287227  -1.197401187
  C   -1.305157887  -2.406452438  -1.455258891
  H   -1.720851041  -2.757172740  -2.435765290
  H   -0.193827189  -2.405006724  -1.532094471
  C   -1.757096912  -3.295747094  -0.295528033
  H   -1.169086166  -3.132514842   0.639455830
  O   -3.126348144  -2.984030963   0.094478020
  C   -3.992592415  -4.145575356  -0.008430080
  H   -4.560219947  -4.153490453   0.954858661
  C   -1.764366960  -4.783913282  -0.749953538
  H   -1.695548774  -4.905330388  -1.867026914
  C   -3.090925126  -5.357638597  -0.226907568
  H   -3.520461164  -6.099752834  -0.933037409
  H   -2.920422284  -5.908068503   0.724935089
  O   -0.674017055  -5.428164883  -0.060804540
  O   -3.123664163  -0.970687548  -3.404274828
  O   -4.124584366  -0.594093229  -0.995428306
  N   -4.928197642  -3.827448368  -1.128680661
  C   -4.491297182  -3.907741052  -2.490721626
  N   -5.215666383  -3.128533732  -3.417578350
  C   -6.262370615  -2.193089395  -3.088686326
  C   -6.771973936  -2.337361456  -1.712399779
  C   -6.109856703  -3.097390068  -0.808168428
  O   -3.545196053  -4.582172636  -2.853355943
  H   -4.886994559  -3.158889824  -4.397242513
  O   -6.554980872  -1.400473174  -3.955322320
  C   -8.019355932  -1.602458850  -1.373847604
  H   -7.898930665  -0.973596361  -0.477515062
  H   -8.338468458  -0.927374061  -2.187556623
  H   -8.863039895  -2.287780710  -1.187001965
  H   -6.458141475  -3.196726495   0.229475383
  P    0.381880156  -6.230068625  -1.034012331
  O    1.645020099  -5.198890827  -1.049464363
  C    2.792605535  -5.643974174  -1.833195075
  H    2.546849185  -6.420121346  -2.591410653
  H    3.082311174  -4.715091027  -2.365349438
  C    3.868199487  -6.122834451  -0.856654892
  H    3.534188245  -5.991992965   0.202468119
  O    4.034084964  -7.561882139  -0.990229800
  C    5.273365842  -7.890822619  -1.636995633
  H    5.781471446  -8.586625129  -0.921415450
  C    5.263060780  -5.495700729  -1.134692590
  H    5.219087545  -4.532643268  -1.694030671
  C    6.042301021  -6.591567391  -1.886532662
  H    6.112207812  -6.341813152  -2.967301668
  H    7.092327705  -6.647304128  -1.539557107
  O    5.892173494  -5.102449598   0.069744884
  O   -0.126374483  -6.574097306  -2.382996919
  O    0.855951462  -7.435157891  -0.054698846
  N    4.940337723  -8.662163210  -2.883639550
  C    4.314873429  -8.048419406  -4.047366380
  N    4.013474302  -8.866381332  -5.132113613
  C    4.274020530 -10.211845550  -5.117680087
  C    4.856062676 -10.827470668  -3.950217921
  C    5.153844312 -10.044085547  -2.871388327
  O    4.076346865  -6.850693319  -4.066492817
  N    3.950613242 -10.921273151  -6.235623992
  H    4.122939364 -11.913651903  -6.306779983
  H    3.544708463 -10.466487723  -7.056324246
  H    0.274011361  -5.884677830  -4.127445922
  H    5.585587464 -10.476986737  -1.954703556
  H    5.050672024 -11.900031846  -3.931008537
  H    6.026187811  -5.851694081   0.692075627
  H   -4.129336944  -1.561970622  -4.842131936
  H   -4.105601974  -1.461761991  -0.434533572
  H    0.406783165  -8.321874083  -0.135126307
  H    3.199740567  -8.062769769  -6.388517551
  H   -2.937997492   1.283505918  -5.931885214
