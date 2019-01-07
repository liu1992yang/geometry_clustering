%mem=64gb
%nproc=28       
%Chk=snap_98.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_98 

2     1 
  O    2.777619868  -4.922562894  -6.809804634
  C    3.231367848  -3.624248404  -6.440153129
  H    4.317289465  -3.676135064  -6.675855833
  H    3.100871226  -3.483091737  -5.337592944
  C    2.565474986  -2.474696349  -7.197705914
  H    2.987041078  -1.492177920  -6.867278681
  O    2.954043345  -2.527876981  -8.610999292
  C    1.820285504  -2.770769252  -9.431222683
  H    2.030133342  -2.271804860 -10.407480729
  C    1.011129719  -2.461750463  -7.175230389
  H    0.552651879  -3.375305290  -6.722548792
  C    0.602145967  -2.283798066  -8.649754298
  H    0.398778046  -1.210828084  -8.862381156
  H   -0.348018760  -2.807525160  -8.894197373
  O    0.570086150  -1.274080681  -6.492133433
  N    1.813189734  -4.267625807  -9.678818298
  C    0.895852524  -5.241968705  -9.212196944
  C    1.536463792  -6.509626496  -9.356857906
  N    2.820710948  -6.292793363  -9.874411892
  C    2.992199380  -4.957104546 -10.040359738
  N   -0.347272016  -5.045692315  -8.699070353
  C   -0.976706210  -6.181798664  -8.226609989
  N   -0.406947922  -7.466693298  -8.330285940
  C    0.907883185  -7.728030451  -8.889597948
  N   -2.251654432  -6.053210418  -7.724600697
  H   -2.626532117  -5.108706179  -7.644541547
  H   -2.560573151  -6.675759610  -6.963521129
  O    1.323974178  -8.857843903  -8.886517655
  H    3.894593644  -4.449007720 -10.394015246
  H   -0.885931227  -8.269076176  -7.889061660
  P    0.378304498  -1.440982363  -4.835565814
  O   -0.060944078   0.154840808  -4.748893389
  C    0.680548844   1.000761167  -3.831747036
  H    1.404381982   1.578384632  -4.448126913
  H    1.232353617   0.406473590  -3.068095461
  C   -0.371306001   1.907007102  -3.199285375
  H   -0.011182210   2.325484710  -2.226289157
  O   -0.562299494   3.061993007  -4.084561667
  C   -1.928369509   3.503358932  -4.033747986
  H   -1.901677787   4.568551401  -3.688422912
  C   -1.793558546   1.295444845  -3.069397762
  H   -2.035526011   0.573585310  -3.887970398
  C   -2.704882545   2.526540608  -3.136063972
  H   -2.867356738   2.939547054  -2.112240184
  H   -3.724241150   2.290077526  -3.495978687
  O   -1.919401373   0.695382461  -1.761056285
  O    1.586972947  -1.892832786  -4.124502022
  O   -0.935372190  -2.390764775  -4.937470886
  N   -2.444357838   3.499693181  -5.441416083
  C   -2.144200545   2.661080803  -6.510122546
  C   -2.992795632   3.050963504  -7.599835646
  N   -3.801353398   4.125013429  -7.200854183
  C   -3.477484346   4.387728993  -5.936905217
  N   -1.238452791   1.619676777  -6.664662468
  C   -1.234165631   0.953600457  -7.893021311
  N   -2.001677486   1.273399302  -8.929412934
  C   -2.908828491   2.349397374  -8.845482749
  H   -0.524804975   0.087925455  -7.983763098
  N   -3.644293250   2.652129417  -9.930071000
  H   -3.543361566   2.139782782 -10.803822825
  H   -4.312142978   3.423046251  -9.910835461
  H   -3.906001324   5.166557198  -5.310100652
  P   -1.617778726  -0.917268182  -1.799181865
  O   -1.232939772  -1.279154293  -0.269392845
  C   -0.739184301  -2.650412618  -0.151279172
  H   -0.327396843  -3.048146781  -1.109193750
  H    0.126683151  -2.541941847   0.556789948
  C   -1.811000584  -3.544630515   0.464957453
  H   -2.063040285  -3.239521408   1.509033127
  O   -3.094013247  -3.421656833  -0.248860714
  C   -3.689013678  -4.712446347  -0.476461921
  H   -4.714776723  -4.623737349  -0.036210780
  C   -1.436622811  -5.047786136   0.341461885
  H   -0.756363134  -5.232797911  -0.539619286
  C   -2.784660698  -5.760904400   0.168042531
  H   -2.695612810  -6.703294041  -0.400793072
  H   -3.164684772  -6.076994814   1.170103311
  O   -0.893665711  -5.507345759   1.588152160
  O   -0.599619410  -1.332874572  -2.828579829
  O   -3.058926141  -1.558406511  -2.163351374
  N   -3.827929303  -4.897922356  -1.966117492
  C   -2.686899517  -4.939818642  -2.822666299
  N   -2.859622963  -5.575282876  -4.068337132
  C   -4.067510503  -6.217807490  -4.511553446
  C   -5.226591036  -6.002371777  -3.631538176
  C   -5.078634087  -5.393254294  -2.430201524
  O   -1.613670169  -4.429518546  -2.531288623
  H   -2.025196536  -5.687779195  -4.672613062
  O   -4.002162052  -6.850845072  -5.547479130
  C   -6.537329483  -6.498428664  -4.129557269
  H   -6.633297622  -6.382630706  -5.223625343
  H   -6.663892478  -7.575628403  -3.922679016
  H   -7.397050725  -5.977983667  -3.682631386
  H   -5.929026307  -5.240085665  -1.748615962
  P    0.733328829  -5.768138844   1.608123593
  O    0.851134674  -6.296577917   0.046058014
  C    2.134878145  -6.706648322  -0.461485250
  H    2.896562647  -6.816170070   0.334222730
  H    1.936344024  -7.714337483  -0.889330818
  C    2.575278802  -5.726062306  -1.558533657
  H    3.526336083  -5.204444027  -1.301113149
  O    2.918323607  -6.579767427  -2.708534456
  C    2.296342412  -6.119668064  -3.908099518
  H    3.088206296  -6.155641791  -4.697748068
  C    1.517774527  -4.719642284  -2.077929734
  H    0.481332848  -4.983614158  -1.742017516
  C    1.681582650  -4.758806122  -3.602289064
  H    0.722705573  -4.557669821  -4.119444249
  H    2.362494787  -3.935140357  -3.947550447
  O    1.821479175  -3.435422204  -1.541776078
  O    1.313194957  -6.573183035   2.670323405
  O    1.251828788  -4.210420704   1.485787123
  N    1.224021746  -7.129968230  -4.287232425
  C    0.184713034  -6.751595329  -5.220399486
  N   -0.872912382  -7.610247146  -5.431265786
  C   -0.896908353  -8.845356464  -4.828230635
  C    0.195008616  -9.283754230  -4.004513930
  C    1.221476423  -8.410086277  -3.743849529
  O    0.241614979  -5.660014650  -5.793062090
  N   -2.012117813  -9.593393579  -5.077795600
  H   -2.128034656 -10.526689989  -4.710615719
  H   -2.795263095  -9.181985930  -5.578333099
  H    1.802755776  -5.048978370  -6.661845144
  H    2.080565288  -8.675564279  -3.102074495
  H    0.211107096 -10.294000082  -3.595964664
  H    1.648297387  -2.717068475  -2.220986377
  H   -1.312569227  -2.749356584  -4.062617087
  H   -3.306350368  -2.436809035  -1.679208587
  H    1.982259281  -3.902870501   2.085687151
  H    3.486514432  -7.037154826 -10.069208100
  H   -0.594666698   1.346211282  -5.888950902

