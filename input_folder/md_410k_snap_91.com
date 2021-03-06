%mem=64gb
%nproc=28       
%Chk=snap_91.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_91 

2     1 
  O    2.855129846  -4.763597180  -7.587201252
  C    2.478803052  -3.540632416  -8.226736633
  H    3.011830216  -3.614226188  -9.204323010
  H    2.882028317  -2.689451649  -7.644695635
  C    0.977489743  -3.366747610  -8.463611757
  H    0.782140214  -2.362946194  -8.923399844
  O    0.570980401  -4.321256568  -9.503416048
  C   -0.622943651  -4.999237674  -9.137281770
  H   -1.241588302  -5.088465717 -10.061619974
  C    0.008184720  -3.591054555  -7.273581173
  H    0.446256124  -4.213583728  -6.458986340
  C   -1.216124373  -4.250453150  -7.938834059
  H   -1.949992569  -3.483271561  -8.267882362
  H   -1.797755234  -4.897124223  -7.248543201
  O   -0.273846129  -2.267980286  -6.790584707
  N   -0.229091319  -6.396140034  -8.706526048
  C   -1.127255194  -7.471508614  -8.513622776
  C   -0.472561548  -8.394831744  -7.647822523
  N    0.811106212  -7.893338408  -7.368243264
  C    0.939863926  -6.692852175  -7.979679088
  N   -2.367083133  -7.655576951  -9.061449088
  C   -3.013897437  -8.786694729  -8.634808699
  N   -2.459769464  -9.719405869  -7.721526727
  C   -1.150657954  -9.544532665  -7.113497437
  N   -4.293710972  -8.998764881  -9.099382671
  H   -4.662223223  -8.369410009  -9.811301470
  H   -4.753603874  -9.893721601  -9.015057448
  O   -0.797501125 -10.331596582  -6.262910572
  H    1.855852551  -5.993353125  -7.964120129
  H   -3.017364678 -10.513658218  -7.379209055
  P   -1.016861527  -2.124497320  -5.312787111
  O   -1.883123983  -0.761002514  -5.686407101
  C   -1.311559584   0.487747582  -5.268529442
  H   -1.710782363   1.198974257  -6.023411432
  H   -0.205762176   0.494322873  -5.317636848
  C   -1.831327686   0.812066934  -3.855514516
  H   -1.161903072   0.434785761  -3.043131540
  O   -1.753559079   2.258964223  -3.668590221
  C   -3.055223278   2.850959573  -3.602476634
  H   -3.027065276   3.518740661  -2.701579504
  C   -3.319680307   0.412160569  -3.682598795
  H   -3.690552438  -0.242983544  -4.512569906
  C   -4.097197895   1.736344898  -3.576285935
  H   -4.672406816   1.738361701  -2.615404885
  H   -4.865476815   1.808590803  -4.368647084
  O   -3.567495124  -0.236346059  -2.424197965
  O   -0.023403155  -1.975320644  -4.224240218
  O   -2.064732143  -3.351217850  -5.485116245
  N   -3.157692389   3.724603631  -4.824338459
  C   -2.088025086   4.261921624  -5.531881906
  C   -2.640184857   5.127386413  -6.531543766
  N   -4.038129719   5.121189630  -6.431290449
  C   -4.339432777   4.300956278  -5.424217146
  N   -0.710866638   4.081895833  -5.404239772
  C    0.092424533   4.813848141  -6.297117272
  N   -0.371712419   5.625461660  -7.233402737
  C   -1.757166901   5.842915691  -7.403383171
  H    1.194222167   4.688638547  -6.204504290
  N   -2.152588412   6.699627073  -8.356822386
  H   -1.479192628   7.193234009  -8.944880429
  H   -3.142993772   6.892766363  -8.513113170
  H   -5.337271511   4.074742525  -5.057036994
  P   -2.857086942  -1.713376380  -2.183524349
  O   -1.679155982  -1.153441346  -1.202137698
  C   -0.515098885  -2.014573868  -1.075535043
  H   -0.138506312  -2.296765100  -2.089354386
  H    0.230939217  -1.361007720  -0.574640762
  C   -0.957338264  -3.189963763  -0.212014391
  H   -1.189950175  -2.895436673   0.838228670
  O   -2.251547797  -3.554650498  -0.815143349
  C   -2.339117676  -5.000970339  -1.026478787
  H   -3.328460627  -5.262033818  -0.573882544
  C   -0.095214540  -4.465695782  -0.268817897
  H    0.593935140  -4.463654160  -1.170531071
  C   -1.130927918  -5.605677499  -0.317110854
  H   -0.728226445  -6.519684307  -0.805083708
  H   -1.394742141  -5.943027865   0.710968593
  O    0.688086760  -4.482110249   0.931310918
  O   -2.428711470  -2.321082883  -3.483100962
  O   -4.115645946  -2.409388560  -1.434176258
  N   -2.388065060  -5.263576428  -2.488874403
  C   -1.202988592  -5.521891069  -3.256757286
  N   -1.400461581  -6.077890689  -4.537153148
  C   -2.677084069  -6.279304020  -5.154779345
  C   -3.825012290  -5.785477653  -4.371924933
  C   -3.662873494  -5.321924530  -3.109426744
  O   -0.081862081  -5.307131092  -2.833486229
  H   -0.537243096  -6.361790494  -5.045636258
  O   -2.687502471  -6.804853787  -6.248431368
  C   -5.144820167  -5.818322277  -5.057918054
  H   -5.223093197  -6.684331412  -5.738918461
  H   -5.993749772  -5.885242130  -4.363540177
  H   -5.294368090  -4.920043726  -5.679645147
  H   -4.506534385  -4.954312593  -2.506215285
  P    1.899410868  -5.626660819   0.908846582
  O    2.765007330  -5.231018532  -0.424090673
  C    3.697249601  -4.117242104  -0.349733308
  H    3.374821363  -3.358384373   0.392377342
  H    4.675357860  -4.554570240  -0.050168037
  C    3.776298153  -3.516853815  -1.753528459
  H    4.464138677  -2.634616397  -1.759288925
  O    4.428496306  -4.491379499  -2.634483065
  C    3.654112560  -4.718246784  -3.814618328
  H    4.402065168  -4.783936978  -4.640099237
  C    2.426816185  -3.186831315  -2.430621655
  H    1.574725321  -3.752213018  -1.968490560
  C    2.622698483  -3.593382905  -3.897324587
  H    1.654086464  -3.892112313  -4.353817293
  H    2.989787367  -2.748382745  -4.509490525
  O    2.184057411  -1.792007848  -2.270611618
  O    1.457804499  -7.019241476   0.907838469
  O    2.805679880  -5.024578283   2.121838029
  N    2.983381449  -6.072869766  -3.688718600
  C    1.992360902  -6.469029632  -4.654164585
  N    1.475535240  -7.742596536  -4.624823610
  C    1.765508893  -8.583190286  -3.571059929
  C    2.621225802  -8.137871983  -2.505574598
  C    3.220422652  -6.906847933  -2.593708493
  O    1.618246612  -5.666532254  -5.525150772
  N    1.201040059  -9.821651945  -3.635327146
  H    1.355629088 -10.517148544  -2.920114447
  H    0.605912748 -10.084509865  -4.424276349
  H    2.597627375  -4.774916252  -6.615598374
  H    3.914524052  -6.523483315  -1.821143365
  H    2.783289582  -8.762694229  -1.624928683
  H    1.524249043  -1.494664358  -2.963172884
  H   -2.641971301  -3.560497946  -4.651516314
  H   -4.029453491  -2.574974442  -0.440738520
  H    2.725274193  -5.430844726   3.030234254
  H    1.406429910  -8.271688637  -6.621479702
  H   -0.349045972   3.465415784  -4.657924223

