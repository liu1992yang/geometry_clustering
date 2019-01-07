%mem=64gb
%nproc=28       
%Chk=snap_113.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_113 

2     1 
  O    3.339480958  -4.287625010  -6.220746747
  C    3.371151928  -3.162361129  -7.092864033
  H    3.935663707  -3.422424806  -8.009819972
  H    3.962814533  -2.420372982  -6.510892327
  C    1.981053967  -2.619843313  -7.445856423
  H    2.006901471  -1.550283834  -7.764143636
  O    1.524869367  -3.314275059  -8.656583583
  C    0.418017640  -4.152965864  -8.381583094
  H   -0.319287283  -3.990036473  -9.207401501
  C    0.917362964  -2.873169979  -6.358335423
  H    1.397944171  -3.241385210  -5.401633039
  C   -0.067907047  -3.883278174  -6.958541985
  H   -1.116726150  -3.507568893  -6.963436325
  H   -0.088052167  -4.810266893  -6.329146586
  O    0.288043149  -1.593534140  -6.115497471
  N    0.912153608  -5.581620939  -8.517324626
  C    0.159666699  -6.725793459  -8.180063381
  C    1.010183042  -7.851376825  -8.387377145
  N    2.263465439  -7.372934566  -8.788304958
  C    2.196820288  -6.012946973  -8.872727163
  N   -1.150586805  -6.765819143  -7.791896865
  C   -1.652246857  -8.033110835  -7.602163063
  N   -0.879099596  -9.203728860  -7.765412607
  C    0.549531607  -9.186600876  -8.066205111
  N   -2.963813845  -8.130957402  -7.213965355
  H   -3.541356599  -7.271785080  -7.202475655
  H   -3.458764672  -9.010786718  -7.189294772
  O    1.131199262 -10.243367385  -8.056946929
  H    2.991096886  -5.333898012  -9.194245885
  H   -1.282807783 -10.124364806  -7.546599939
  P   -0.628490751  -1.559355624  -4.732224701
  O   -0.877107255   0.084629648  -4.742628185
  C   -0.174042291   0.820121113  -3.708475608
  H    0.445074556   1.562989500  -4.263354959
  H    0.496214298   0.171743891  -3.101031771
  C   -1.208648942   1.525202766  -2.829317613
  H   -0.906130520   1.509045727  -1.749775397
  O   -1.159903846   2.936922530  -3.199888508
  C   -2.472650128   3.493461101  -3.312736606
  H   -2.499286182   4.366950209  -2.610854366
  C   -2.685514405   1.079128105  -2.998603551
  H   -2.868193971   0.466958247  -3.914876735
  C   -3.488631950   2.387367385  -3.016036132
  H   -3.965635192   2.533359136  -2.014504645
  H   -4.341515065   2.359166369  -3.718930677
  O   -3.095012837   0.367951806  -1.822151763
  O    0.126687159  -2.070186702  -3.566646990
  O   -1.969044662  -2.178957576  -5.407680328
  N   -2.582361185   4.036385781  -4.702543338
  C   -2.355393894   3.411104237  -5.925739574
  C   -2.515423185   4.414786267  -6.938918935
  N   -2.820661440   5.645780670  -6.339491834
  C   -2.847078051   5.425008091  -5.028171123
  N   -2.049706546   2.098342205  -6.273264161
  C   -1.857586312   1.838148261  -7.640410766
  N   -1.999374094   2.731588861  -8.607764143
  C   -2.336311537   4.070924183  -8.316178441
  H   -1.578787260   0.793681486  -7.910122739
  N   -2.467021356   4.931999074  -9.338767376
  H   -2.321888985   4.637952729 -10.304231373
  H   -2.708527791   5.910623459  -9.175280279
  H   -3.047057717   6.163642938  -4.254232717
  P   -2.788289465  -1.265645551  -1.842029501
  O   -1.678524190  -1.269387495  -0.656032848
  C   -0.767357105  -2.407633737  -0.647885484
  H   -0.276140945  -2.521307040  -1.646543649
  H   -0.012631351  -2.098060283   0.109427210
  C   -1.588762079  -3.613160058  -0.206911060
  H   -1.858169568  -3.576045368   0.877538339
  O   -2.855874599  -3.432393076  -0.931840720
  C   -3.323125764  -4.701707805  -1.493329301
  H   -4.369842231  -4.779457947  -1.103759100
  C   -1.083634451  -5.020421050  -0.599562436
  H   -0.356043161  -5.008444561  -1.448175716
  C   -2.373487431  -5.776871338  -0.973716421
  H   -2.172191773  -6.603053089  -1.685040465
  H   -2.787830045  -6.282628437  -0.071255165
  O   -0.566437402  -5.609006283   0.602811270
  O   -2.362045358  -1.669864139  -3.224694611
  O   -4.280885314  -1.733481335  -1.397557962
  N   -3.367254832  -4.565733190  -2.972225087
  C   -2.321456548  -5.063534222  -3.818873333
  N   -2.690922353  -5.295323272  -5.162987998
  C   -3.980233381  -4.994226911  -5.724283329
  C   -4.832570307  -4.158501154  -4.859748315
  C   -4.536719189  -4.001058661  -3.547796078
  O   -1.200516615  -5.307567735  -3.419035237
  H   -1.998123376  -5.779745564  -5.765975351
  O   -4.233377458  -5.464159085  -6.813623835
  C   -6.032967073  -3.560307814  -5.501036800
  H   -6.677555771  -4.343572932  -5.942524885
  H   -6.661663048  -2.988038628  -4.805284894
  H   -5.755873704  -2.886800909  -6.328316222
  H   -5.176966440  -3.425267763  -2.859643951
  P    0.931948545  -6.293378178   0.466734549
  O    1.784138304  -4.906152761   0.335741361
  C    3.193778029  -5.017537415  -0.000232925
  H    3.661147348  -4.321176126   0.732886165
  H    3.601988803  -6.036066503   0.151295708
  C    3.418992062  -4.545157432  -1.439274848
  H    4.372881638  -3.964256504  -1.526393506
  O    3.638042370  -5.737010620  -2.253630506
  C    2.913395985  -5.634520832  -3.496006235
  H    3.685006882  -5.442023523  -4.287897377
  C    2.247090741  -3.762381131  -2.070392538
  H    1.396204743  -3.615817116  -1.373479258
  C    1.867496106  -4.532540897  -3.336921026
  H    0.829766311  -4.925934298  -3.301984022
  H    1.887766733  -3.860457964  -4.229143593
  O    2.767409513  -2.464512831  -2.386321250
  O    1.112180999  -7.327400321  -0.563566091
  O    1.197287579  -6.727442172   2.009462841
  N    2.292108348  -6.975782379  -3.767990728
  C    1.618968924  -7.175150246  -5.045844103
  N    1.218280655  -8.443568614  -5.394258836
  C    1.397274676  -9.484046476  -4.510362205
  C    1.930405034  -9.272065129  -3.200076395
  C    2.369589078  -8.016541920  -2.850622090
  O    1.462150269  -6.195349735  -5.780846123
  N    0.998143331 -10.726360314  -4.949716078
  H    1.253375273 -11.558366975  -4.437500978
  H    0.784057603 -10.853246805  -5.933582338
  H    2.693241000  -4.985509142  -6.504309844
  H    2.806276039  -7.788488760  -1.859510131
  H    1.974313313 -10.081816510  -2.473016948
  H    2.075647292  -1.967215231  -2.894207685
  H   -2.723323517  -2.439635519  -4.750427500
  H   -4.406248943  -2.033161138  -0.441835811
  H    1.063767061  -7.683859437   2.266071964
  H    3.066466936  -7.967230017  -8.976770695
  H   -1.850679281   1.370951172  -5.555594053
