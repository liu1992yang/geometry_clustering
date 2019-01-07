%mem=64gb
%nproc=28       
%Chk=snap_125.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_125 

2     1 
  O    2.808177717  -4.786624056  -7.432782879
  C    2.503671676  -3.523268877  -8.029087823
  H    3.081006614  -3.572835676  -8.983291936
  H    2.906319767  -2.714765564  -7.388851675
  C    1.021902739  -3.290784230  -8.326680292
  H    0.877916823  -2.266386371  -8.759223037
  O    0.630936299  -4.196056273  -9.413625916
  C   -0.604745727  -4.838999202  -9.125905636
  H   -1.184598575  -4.861598552 -10.079671768
  C   -0.003278604  -3.523300165  -7.186604835
  H    0.379542237  -4.186423843  -6.378309907
  C   -1.219351130  -4.121872472  -7.920319502
  H   -1.924482602  -3.324907196  -8.238971798
  H   -1.835094145  -4.784536539  -7.272088532
  O   -0.262263742  -2.209842804  -6.670800064
  N   -0.282433679  -6.267006602  -8.744794977
  C   -1.212550338  -7.328549110  -8.678985300
  C   -0.615794785  -8.343412386  -7.876214848
  N    0.652736756  -7.889587653  -7.476276193
  C    0.838348701  -6.648364836  -7.985940751
  N   -2.433088445  -7.421981913  -9.290999335
  C   -3.116666825  -8.578591811  -9.018599556
  N   -2.621981524  -9.609457194  -8.183333158
  C   -1.329597282  -9.540888641  -7.513285936
  N   -4.371522739  -8.716844381  -9.570889625
  H   -4.692358887  -8.010354645 -10.231390642
  H   -4.852848462  -9.604316864  -9.597292483
  O   -1.014998358 -10.441388515  -6.773114074
  H    1.763618501  -5.980817661  -7.881911629
  H   -3.202618877 -10.430386818  -7.963822612
  P   -1.058125475  -2.096631777  -5.215663197
  O   -1.886261541  -0.710679234  -5.587936818
  C   -1.303168194   0.520662505  -5.134710585
  H   -1.601807695   1.236013756  -5.930520477
  H   -0.197766479   0.483517301  -5.078886692
  C   -1.936745560   0.885061836  -3.779440602
  H   -1.337083415   0.525322803  -2.904442039
  O   -1.861459669   2.336629892  -3.624314123
  C   -3.159396544   2.937063602  -3.670746788
  H   -3.170998695   3.676691702  -2.828141095
  C   -3.437744113   0.498743707  -3.711613966
  H   -3.765563274  -0.123718731  -4.583864866
  C   -4.205166321   1.829639395  -3.603704758
  H   -4.748997156   1.847203381  -2.623630852
  H   -5.001465445   1.895295865  -4.366909577
  O   -3.772508383  -0.192192910  -2.497421245
  O   -0.094010385  -2.002000722  -4.095575198
  O   -2.107364962  -3.309765710  -5.459111203
  N   -3.193318726   3.706859127  -4.964795515
  C   -2.078534377   4.192280121  -5.639930410
  C   -2.561634443   4.931305711  -6.767847980
  N   -3.962820183   4.901066646  -6.777927506
  C   -4.331568555   4.184795538  -5.715040253
  N   -0.713410732   4.063951477  -5.382978609
  C    0.148471629   4.720227315  -6.279532503
  N   -0.251270424   5.411653406  -7.334726887
  C   -1.621445614   5.566978417  -7.641676143
  H    1.241091231   4.640082793  -6.081433732
  N   -1.951316740   6.291925720  -8.721138964
  H   -1.239981855   6.732976250  -9.306131827
  H   -2.928538039   6.439350001  -8.978390843
  H   -5.352949371   3.976832801  -5.406508006
  P   -3.017927051  -1.637602266  -2.204398719
  O   -1.865263248  -0.991291341  -1.243337280
  C   -0.692382509  -1.815535269  -1.010657204
  H   -0.239287873  -2.125426290  -1.984144573
  H    0.001557816  -1.124168620  -0.487414063
  C   -1.157983914  -2.966151987  -0.129360162
  H   -1.480375821  -2.632921178   0.886334366
  O   -2.383625450  -3.428841527  -0.797986724
  C   -2.380239433  -4.890128900  -0.941411829
  H   -3.379040964  -5.180578191  -0.527868880
  C   -0.230343074  -4.191710642  -0.045158859
  H    0.532191974  -4.199388876  -0.878335446
  C   -1.191143223  -5.392897442  -0.129329876
  H   -0.692770874  -6.293989017  -0.553032771
  H   -1.507797117  -5.711851707   0.889835722
  O    0.406547684  -4.100483050   1.233121839
  O   -2.547764875  -2.281694346  -3.470473194
  O   -4.269001135  -2.341948918  -1.453868055
  N   -2.344800885  -5.221565079  -2.390516075
  C   -1.120649654  -5.489345471  -3.091815002
  N   -1.248614650  -6.003729702  -4.397905767
  C   -2.488460758  -6.205344865  -5.086313735
  C   -3.681099155  -5.756711264  -4.345205002
  C   -3.586351805  -5.303221266  -3.071931407
  O   -0.024786704  -5.312534170  -2.590266735
  H   -0.360284519  -6.257025286  -4.893063810
  O   -2.435480220  -6.688909717  -6.198603391
  C   -4.970937204  -5.831913024  -5.082377205
  H   -5.170820244  -6.862101407  -5.424089541
  H   -5.835790595  -5.517284334  -4.483494225
  H   -4.950504219  -5.201528688  -5.987103456
  H   -4.466951594  -4.965813237  -2.504496139
  P    1.768962853  -5.047396716   1.383191790
  O    2.818295168  -4.145624311   0.479426288
  C    3.511313882  -4.858174007  -0.571835548
  H    4.512652819  -5.124716130  -0.168059061
  H    2.979764735  -5.783348424  -0.879556679
  C    3.650931953  -3.856457706  -1.727597508
  H    4.273430004  -2.972793977  -1.454124971
  O    4.432579904  -4.576180467  -2.741233492
  C    3.677954000  -4.776045593  -3.934759488
  H    4.417686179  -4.660047229  -4.764019882
  C    2.334833558  -3.436261334  -2.414667640
  H    1.437929092  -3.951444113  -1.976445715
  C    2.520010116  -3.784547349  -3.901698379
  H    1.564793827  -4.168058502  -4.319481126
  H    2.756844003  -2.874143198  -4.487228604
  O    2.173731925  -2.032408954  -2.219754258
  O    1.635299040  -6.451541477   0.992782629
  O    2.177690656  -4.662358437   2.899106958
  N    3.213799206  -6.226610904  -3.935100967
  C    2.102739609  -6.665157850  -4.739437358
  N    1.746811135  -7.991229335  -4.767965021
  C    2.405760576  -8.905035717  -3.975455229
  C    3.507035870  -8.485592988  -3.147135794
  C    3.888494145  -7.170265787  -3.156366742
  O    1.488376918  -5.838198226  -5.438435935
  N    1.975478774 -10.194333468  -4.063540084
  H    2.386661537 -10.934948835  -3.512680414
  H    1.176380617 -10.439536954  -4.644003288
  H    2.486518415  -4.853952921  -6.487786372
  H    4.743759324  -6.799552651  -2.562061982
  H    4.035389393  -9.208077768  -2.523165689
  H    1.469102562  -1.694742120  -2.843981326
  H   -2.717349596  -3.534942925  -4.652651082
  H   -4.204044325  -2.470167998  -0.454557921
  H    2.339622177  -3.693842943   3.109699324
  H    1.224696709  -8.347804966  -6.757055718
  H   -0.402422060   3.529726221  -4.554875542
