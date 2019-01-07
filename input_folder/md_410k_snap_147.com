%mem=64gb
%nproc=28       
%Chk=snap_147.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_147 

2     1 
  O    0.956907052  -3.472543393  -4.738196180
  C    1.628389714  -3.569810477  -5.986707351
  H    1.947453436  -4.641248367  -6.025338342
  H    2.532810019  -2.933914761  -5.981995747
  C    0.693829345  -3.203455441  -7.152280674
  H    0.856485306  -2.162659383  -7.514827431
  O    1.078355955  -4.037836273  -8.295044843
  C   -0.034784205  -4.762387092  -8.807248927
  H    0.042971762  -4.725772433  -9.920671866
  C   -0.809240363  -3.482002276  -6.898100207
  H   -0.980654840  -4.076102156  -5.970655422
  C   -1.301509262  -4.196460886  -8.163549513
  H   -1.818268517  -3.474924872  -8.835221529
  H   -2.076476810  -4.968567944  -7.946210192
  O   -1.498420922  -2.211267308  -6.857947282
  N    0.174669124  -6.199430683  -8.381092685
  C   -0.699652617  -7.276211749  -8.641692319
  C   -0.300477992  -8.338583038  -7.776142894
  N    0.810357685  -7.908202988  -7.041200157
  C    1.070999640  -6.624370124  -7.380617655
  N   -1.716507614  -7.344778412  -9.553460681
  C   -2.426066193  -8.519829671  -9.525796338
  N   -2.144262750  -9.592062947  -8.641279561
  C   -1.075775933  -9.546179807  -7.657897302
  N   -3.485855998  -8.639030058 -10.391179572
  H   -3.664593823  -7.894570396 -11.062884132
  H   -3.978853804  -9.508319143 -10.533813121
  O   -0.966786611 -10.460995625  -6.873429655
  H    1.873285947  -5.987846576  -6.950004315
  H   -2.747657155 -10.426327240  -8.617483068
  P   -1.907922611  -1.673951021  -5.355695840
  O   -2.792363478  -0.357826474  -5.779832365
  C   -2.048105717   0.725253309  -6.380159151
  H   -2.865992804   1.427067675  -6.654720660
  H   -1.549186202   0.385796109  -7.310780015
  C   -1.057325520   1.370136700  -5.398912909
  H   -0.039559352   0.900904532  -5.423334026
  O   -0.830400632   2.737271262  -5.874890537
  C   -1.208525973   3.694078269  -4.884425844
  H   -0.422229957   4.488447755  -4.921853135
  C   -1.573992178   1.491072202  -3.940274761
  H   -2.640989523   1.175105461  -3.837023873
  C   -1.347620274   2.964009669  -3.556272308
  H   -0.398459744   3.034292245  -2.960412112
  H   -2.109397568   3.359347707  -2.862385650
  O   -0.671870234   0.739751295  -3.106322541
  O   -0.647529852  -1.346849939  -4.561033329
  O   -2.871109897  -2.883913805  -4.880309432
  N   -2.503739562   4.289077134  -5.362601248
  C   -2.662782362   5.027798984  -6.533041773
  C   -4.049927759   5.373708449  -6.622147387
  N   -4.739772461   4.821335014  -5.533804909
  C   -3.831567223   4.173883187  -4.804354527
  N   -1.752026718   5.437198195  -7.506774815
  C   -2.276666156   6.218010938  -8.557118211
  N   -3.543365220   6.578402844  -8.664524954
  C   -4.507779267   6.184374043  -7.710380572
  H   -1.560918191   6.553658770  -9.342041520
  N   -5.773990013   6.591949634  -7.878613705
  H   -6.046422798   7.175807468  -8.670938237
  H   -6.503850202   6.332266307  -7.212727617
  H   -4.024261369   3.625383264  -3.884993501
  P   -1.336317263  -0.625788624  -2.476919927
  O   -0.097260652  -1.456845146  -1.845019083
  C   -0.048814123  -2.895998144  -1.989579751
  H   -0.419654494  -3.244760005  -2.974652063
  H    1.051781152  -3.071546651  -1.939114928
  C   -0.790028021  -3.539832279  -0.815402412
  H   -0.862664916  -2.854638449   0.063173340
  O   -2.168196201  -3.770721477  -1.228081705
  C   -2.478738818  -5.172718200  -1.236868313
  H   -3.525642540  -5.230074625  -0.851390427
  C   -0.219498923  -4.918336394  -0.386782935
  H    0.590722841  -5.285173405  -1.065231128
  C   -1.431703190  -5.867908613  -0.368817991
  H   -1.159338912  -6.886960779  -0.714705144
  H   -1.805423928  -6.003014696   0.670342716
  O    0.266153452  -4.719273718   0.954326127
  O   -2.495044548  -1.245624460  -3.227788755
  O   -1.921619140   0.079917873  -1.115852586
  N   -2.471449150  -5.614829408  -2.675512656
  C   -1.270050494  -6.014163673  -3.330737462
  N   -1.385279006  -6.399278288  -4.687539419
  C   -2.584466315  -6.308343425  -5.463077363
  C   -3.747806759  -5.781603954  -4.737442425
  C   -3.668073496  -5.472962249  -3.418823478
  O   -0.184003816  -6.052949176  -2.779225029
  H   -0.530495259  -6.797003420  -5.104826416
  O   -2.514759895  -6.649719936  -6.629651633
  C   -4.985422231  -5.557848661  -5.532571491
  H   -5.896115054  -5.863674216  -4.998426350
  H   -5.090658323  -4.490126245  -5.794466019
  H   -4.972134061  -6.117814336  -6.482736508
  H   -4.537308237  -5.099266867  -2.859195551
  P    1.541157917  -5.686130375   1.373892295
  O    2.729601474  -4.604966946   1.029122373
  C    4.064957310  -5.132690566   0.839692511
  H    4.673405561  -4.502359401   1.528349684
  H    4.159300304  -6.201391162   1.126944372
  C    4.503030315  -4.928652871  -0.612232115
  H    5.601109698  -4.716433134  -0.642597990
  O    4.368678545  -6.209625625  -1.296729651
  C    3.824593692  -6.021028109  -2.614302412
  H    4.663107827  -6.211039619  -3.331364356
  C    3.703324730  -3.876347300  -1.426998107
  H    2.884382744  -3.406396757  -0.829575133
  C    3.226758094  -4.612388068  -2.686398925
  H    2.120126370  -4.634816999  -2.786183922
  H    3.534244092  -4.085067779  -3.612842052
  O    4.515219192  -2.756031637  -1.735966817
  O    1.645273003  -7.009568959   0.751484059
  O    1.406065713  -5.631293756   2.988725191
  N    2.798224685  -7.097256749  -2.823735871
  C    2.266970226  -7.232670194  -4.164934706
  N    1.398487451  -8.279233137  -4.430646561
  C    0.992126368  -9.123661792  -3.423676678
  C    1.467549245  -8.948646097  -2.084328128
  C    2.366281265  -7.944943296  -1.809043862
  O    2.605217749  -6.434204530  -5.034125972
  N    0.085531104 -10.087835776  -3.783307234
  H   -0.167081386 -10.830280944  -3.147206979
  H   -0.165328945 -10.224278032  -4.761708558
  H    0.707044746  -2.515705801  -4.521796623
  H    2.783828479  -7.774798933  -0.792568234
  H    1.121645541  -9.594982325  -1.277655787
  H    5.282248526  -2.988629402  -2.303012870
  H   -3.107626760  -2.868583921  -3.867969969
  H   -1.850602602  -0.384103975  -0.250277042
  H    1.266428044  -4.740406122   3.422350942
  H    1.230260984  -8.432460749  -6.220744927
  H   -0.775449736   5.126050821  -7.482304757

