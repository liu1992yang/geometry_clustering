%mem=64gb
%nproc=28       
%Chk=snap_108.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_108 

2     1 
  O    3.066928085  -5.203633599  -6.779054918
  C    3.000744498  -3.898501851  -7.355488066
  H    3.727153911  -3.979508708  -8.198316921
  H    3.365959745  -3.151793820  -6.621978474
  C    1.622426713  -3.500347059  -7.884057175
  H    1.671357248  -2.469734952  -8.321927745
  O    1.299880508  -4.360148760  -9.029549746
  C   -0.030721962  -4.851646585  -8.942365481
  H   -0.456777299  -4.811316794  -9.973686453
  C    0.412802404  -3.606427470  -6.921391475
  H    0.582472412  -4.307131326  -6.069908088
  C   -0.739237205  -4.060562019  -7.837519863
  H   -1.283956214  -3.188278183  -8.254988245
  H   -1.518832345  -4.642733026  -7.296936487
  O    0.222582556  -2.262869067  -6.440505907
  N    0.051964408  -6.306013878  -8.537912394
  C   -0.998131678  -7.242803551  -8.658773879
  C   -0.665449496  -8.339414687  -7.812585502
  N    0.571184425  -8.055780383  -7.202690317
  C    0.988441229  -6.840970692  -7.635713019
  N   -2.105297142  -7.160734314  -9.457989358
  C   -2.960376152  -8.227793953  -9.353425711
  N   -2.727626322  -9.339029209  -8.510486919
  C   -1.556608704  -9.458799793  -7.646225309
  N   -4.116479183  -8.185847919 -10.100734108
  H   -4.239830606  -7.426823725 -10.768332902
  H   -4.711594391  -8.991901252 -10.220444434
  O   -1.477013131 -10.428450720  -6.933383256
  H    1.960682567  -6.295722823  -7.367254180
  H   -3.421125051 -10.095654677  -8.435575008
  P   -0.735131025  -2.058484598  -5.103781721
  O   -1.740516664  -0.876226860  -5.680607355
  C   -1.318403085   0.475566466  -5.439598931
  H   -1.841335379   1.036125684  -6.244016220
  H   -0.225915041   0.605650286  -5.566412312
  C   -1.794186662   0.913474123  -4.042309831
  H   -1.065177318   0.653863767  -3.234003068
  O   -1.783837019   2.376579891  -4.025125755
  C   -3.088883613   2.910118747  -3.767637088
  H   -2.971364386   3.557012976  -2.858447386
  C   -3.245175974   0.461375641  -3.739375720
  H   -3.635556798  -0.262065601  -4.498359913
  C   -4.075743015   1.752667470  -3.638400535
  H   -4.594971513   1.768035598  -2.645945294
  H   -4.888437904   1.765350668  -4.388082272
  O   -3.360958694  -0.095662202  -2.419971110
  O    0.121737199  -1.684170831  -3.950770622
  O   -1.567578228  -3.445356799  -5.217682071
  N   -3.406053468   3.805744402  -4.931833561
  C   -2.511507292   4.321927099  -5.860654120
  C   -3.252935336   5.215691518  -6.701275714
  N   -4.589699096   5.249459680  -6.281715201
  C   -4.676850072   4.426475859  -5.237391793
  N   -1.146934863   4.108039953  -6.051737036
  C   -0.547294994   4.837723875  -7.094833215
  N   -1.191395362   5.670261774  -7.896144280
  C   -2.574454356   5.918167808  -7.748718225
  H    0.543880135   4.687466681  -7.255592176
  N   -3.161253010   6.785123937  -8.587365313
  H   -2.631595725   7.273700784  -9.310284648
  H   -4.158473865   6.994303851  -8.515140271
  H   -5.569492765   4.226131994  -4.649376003
  P   -2.764684329  -1.613646721  -2.157556738
  O   -1.644551085  -1.149863435  -1.067271046
  C   -0.516394633  -2.055483787  -0.924601851
  H   -0.053553126  -2.260497904  -1.919754439
  H    0.192114152  -1.464662365  -0.303334381
  C   -1.065626943  -3.269907107  -0.189006426
  H   -1.371417761  -3.014489468   0.855990992
  O   -2.312565404  -3.576069990  -0.905598348
  C   -2.498049701  -5.034438473  -1.020400468
  H   -3.519634534  -5.190936745  -0.593696199
  C   -0.257399245  -4.582809252  -0.213184402
  H    0.402702853  -4.681007805  -1.121910682
  C   -1.363422288  -5.656302583  -0.204982269
  H   -0.995870794  -6.631826694  -0.593510287
  H   -1.681096710  -5.876819669   0.837876262
  O    0.476251025  -4.620516243   1.015566177
  O   -2.318428805  -2.243591546  -3.439550135
  O   -4.089234707  -2.248761314  -1.466280563
  N   -2.521209661  -5.377345003  -2.466483838
  C   -1.375145931  -5.915632666  -3.142410557
  N   -1.539221634  -6.280862079  -4.490168092
  C   -2.745085624  -6.116566195  -5.248090722
  C   -3.863931402  -5.526117684  -4.486230768
  C   -3.731967313  -5.178839771  -3.182533025
  O   -0.303516376  -6.053467490  -2.576138745
  H   -0.678088863  -6.636938981  -4.963566719
  O   -2.722817040  -6.443861596  -6.415694949
  C   -5.138230994  -5.341290854  -5.233163929
  H   -5.848604676  -6.157461633  -5.017463306
  H   -5.640616072  -4.394014871  -4.993178462
  H   -4.976351732  -5.350498168  -6.326122614
  H   -4.553702525  -4.711355146  -2.618602923
  P    2.068867088  -5.061725355   0.877931337
  O    2.593813452  -3.655913044   0.192516960
  C    3.951675680  -3.634732436  -0.322383037
  H    4.369152477  -2.713275543   0.141107880
  H    4.546798992  -4.515372914  -0.005300767
  C    3.925962658  -3.520293080  -1.849219078
  H    4.809605528  -2.950651557  -2.231349458
  O    4.107430992  -4.866711939  -2.395785304
  C    3.088542127  -5.123782555  -3.376167940
  H    3.569942932  -4.986821870  -4.382245256
  C    2.601339508  -2.957369151  -2.417231314
  H    1.978946003  -2.470333900  -1.637528859
  C    1.932203725  -4.159485972  -3.089942036
  H    1.166926688  -4.633137349  -2.427130797
  H    1.383254880  -3.886697159  -4.004647090
  O    2.962573566  -1.970366474  -3.385888048
  O    2.410335227  -6.302501953   0.184012033
  O    2.509065877  -4.929899131   2.432257677
  N    2.669201798  -6.556380548  -3.271457074
  C    1.751736930  -7.030236053  -4.282536823
  N    1.374395518  -8.346160452  -4.314443585
  C    1.776014618  -9.183103639  -3.290158263
  C    2.595092134  -8.699212357  -2.215859028
  C    3.024998576  -7.394375249  -2.216192778
  O    1.351982377  -6.211259011  -5.133811323
  N    1.349418283 -10.477377092  -3.371907094
  H    1.604057849 -11.165993308  -2.679244188
  H    0.764427613 -10.787489653  -4.139966062
  H    2.627237087  -5.251202186  -5.879267903
  H    3.660334473  -6.963159480  -1.416703814
  H    2.865262217  -9.353707971  -1.386133594
  H    2.126541019  -1.609314645  -3.801906245
  H   -2.227269048  -3.628864754  -4.440280466
  H   -4.039126827  -2.458958994  -0.480839243
  H    2.186206659  -4.139493295   2.954758239
  H    0.976213047  -8.602253314  -6.431727184
  H   -0.636828487   3.493127776  -5.398349920
