%mem=64gb
%nproc=28       
%Chk=snap_101.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_101 

2     1 
  O    2.906275201  -5.182384500  -6.801782715
  C    3.342467536  -3.881668398  -6.423645885
  H    4.374036207  -3.846220681  -6.838734556
  H    3.393925628  -3.820129367  -5.309144213
  C    2.495723482  -2.732586513  -6.974597711
  H    2.904668499  -1.739914738  -6.654028427
  O    2.632568012  -2.683060123  -8.429859096
  C    1.411595596  -3.060649173  -9.056824411
  H    1.384187950  -2.509710340 -10.027733334
  C    0.974018835  -2.817314118  -6.680279791
  H    0.679078175  -3.720648246  -6.100401219
  C    0.295099721  -2.760841090  -8.061562274
  H   -0.104648417  -1.739105333  -8.241108236
  H   -0.591430672  -3.429807321  -8.126600141
  O    0.677266228  -1.590057830  -5.988845860
  N    1.581042590  -4.532989066  -9.375329619
  C    0.787778935  -5.638516207  -8.984497357
  C    1.585931899  -6.805043640  -9.192474627
  N    2.837103326  -6.399977578  -9.677096269
  C    2.841359673  -5.046502786  -9.758067599
  N   -0.483960298  -5.632198478  -8.503881287
  C   -0.978367284  -6.870415143  -8.148415997
  N   -0.233183064  -8.062264890  -8.279201905
  C    1.116393801  -8.119439449  -8.812427837
  N   -2.285825033  -6.952731766  -7.735314570
  H   -2.808375673  -6.069232595  -7.586218666
  H   -2.560779688  -7.701915827  -7.112220408
  O    1.674141424  -9.185357194  -8.860294645
  H    3.675016960  -4.406023405 -10.061777204
  H   -0.610725064  -8.944573391  -7.906695042
  P   -0.263098171  -1.658939413  -4.628536080
  O   -0.551909302  -0.033631599  -4.624898380
  C    0.113402582   0.809890083  -3.657364291
  H    0.737977481   1.502852769  -4.270800286
  H    0.758151912   0.230029098  -2.965911034
  C   -0.967199752   1.586879709  -2.905607107
  H   -0.651913740   1.810548833  -1.854194009
  O   -1.086620419   2.892434912  -3.554333478
  C   -2.457034649   3.295761869  -3.650003562
  H   -2.508647582   4.308692502  -3.173876287
  C   -2.395566863   0.979999220  -2.934688150
  H   -2.566674207   0.307098531  -3.815105417
  C   -3.313604815   2.209776845  -2.990125029
  H   -3.611936838   2.499457641  -1.953455309
  H   -4.273027904   2.014975055  -3.501683078
  O   -2.630212001   0.297463401  -1.707461566
  O    0.429521923  -2.316068569  -3.508018747
  O   -1.629816613  -2.110478810  -5.396850702
  N   -2.765619061   3.466440841  -5.104882242
  C   -2.742968335   2.538084364  -6.142548474
  C   -3.157433015   3.230402066  -7.329048650
  N   -3.407636848   4.576698547  -7.024432023
  C   -3.170233233   4.714181285  -5.722476709
  N   -2.444435366   1.179864195  -6.185368000
  C   -2.547396964   0.549573567  -7.433584118
  N   -2.937451837   1.144825613  -8.552635496
  C   -3.258933691   2.518780003  -8.566291965
  H   -2.300662982  -0.537324697  -7.456902323
  N   -3.648111687   3.069481044  -9.728746848
  H   -3.715035504   2.517648626 -10.582427412
  H   -3.885380219   4.060515703  -9.788214132
  H   -3.258621306   5.632987901  -5.146154917
  P   -2.399426298  -1.355136499  -1.765322765
  O   -1.219369702  -1.373563035  -0.648465371
  C   -0.443106487  -2.608237280  -0.583237175
  H   -0.095389797  -2.905269411  -1.599534683
  H    0.431716133  -2.321308864   0.040253401
  C   -1.353276873  -3.623121617   0.097517242
  H   -1.541490196  -3.383112912   1.171066484
  O   -2.640365229  -3.381826285  -0.571802310
  C   -3.241306262  -4.631150649  -1.020817764
  H   -4.282769997  -4.585981147  -0.614324850
  C   -1.029960130  -5.119088960  -0.112738514
  H   -0.259095949  -5.297528467  -0.917731821
  C   -2.386387394  -5.759933919  -0.451383541
  H   -2.272131005  -6.630411023  -1.126680859
  H   -2.833950467  -6.181057051   0.481522420
  O   -0.640191963  -5.695313374   1.150728561
  O   -2.035742115  -1.815381082  -3.149216893
  O   -3.925539566  -1.706666181  -1.306044173
  N   -3.305745804  -4.582853922  -2.510342035
  C   -2.220257477  -5.024299257  -3.328086918
  N   -2.477374181  -5.068307572  -4.719968676
  C   -3.718243551  -4.692576388  -5.340384558
  C   -4.674508755  -4.036718089  -4.440250738
  C   -4.459876669  -4.008378697  -3.101898587
  O   -1.141595492  -5.367530755  -2.882169833
  H   -1.751253174  -5.538314013  -5.302366893
  O   -3.832491972  -4.916149803  -6.528906458
  C   -5.892948234  -3.456886424  -5.068929150
  H   -5.736114104  -3.227100741  -6.136861578
  H   -6.737859177  -4.167254932  -5.032810161
  H   -6.225082542  -2.529691178  -4.582372567
  H   -5.170638724  -3.534274669  -2.405454076
  P    0.978704121  -5.812452060   1.407569885
  O    1.378359193  -6.503328708  -0.039803145
  C    2.785048499  -6.647338369  -0.340329645
  H    3.439533670  -6.364126170   0.507043787
  H    2.914055092  -7.738222474  -0.529786585
  C    3.118748444  -5.837114060  -1.598815075
  H    4.154476850  -5.417297735  -1.544021423
  O    3.185544303  -6.797044622  -2.708223116
  C    2.421531357  -6.360332752  -3.832524395
  H    3.100961738  -6.482135301  -4.714924811
  C    2.105852672  -4.753773267  -2.025941073
  H    1.134203033  -4.825924205  -1.477199809
  C    1.916803382  -4.951104752  -3.536573493
  H    0.853368683  -4.787774920  -3.821537501
  H    2.503589286  -4.190010589  -4.099870591
  O    2.710009612  -3.490356424  -1.744166197
  O    1.467064569  -6.398281788   2.643092597
  O    1.390777114  -4.234208180   1.118429414
  N    1.266371854  -7.326107290  -4.018924486
  C    0.280270598  -7.051687153  -5.046132105
  N   -0.754794689  -7.944475247  -5.248000117
  C   -0.851879915  -9.059812933  -4.448610225
  C    0.076902695  -9.307627043  -3.385041626
  C    1.116627342  -8.431105187  -3.193855482
  O    0.389211639  -6.017064600  -5.713302644
  N   -1.881975608  -9.918998790  -4.726845762
  H   -2.010103884 -10.775823696  -4.207515798
  H   -2.519691749  -9.742973587  -5.485803619
  H    1.971443193  -5.384588552  -6.519764949
  H    1.873754445  -8.571063721  -2.401266345
  H   -0.029002142 -10.181535916  -2.742574510
  H    2.182393094  -2.773872691  -2.184771788
  H   -2.248814558  -2.728663504  -4.903513137
  H   -4.107810001  -1.738827228  -0.315324626
  H    2.080892340  -3.807893866   1.686047588
  H    3.595195864  -7.042332968  -9.893849594
  H   -2.006831466   0.683193594  -5.369877037
