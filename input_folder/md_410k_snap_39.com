%mem=64gb
%nproc=28       
%Chk=snap_39.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_39 

2     1 
  O    1.616743142  -3.924151073  -6.426589286
  C    1.913757458  -3.539519971  -7.763240501
  H    2.162846741  -4.495595079  -8.266214894
  H    2.813936719  -2.895087947  -7.768032041
  C    0.721588680  -2.847565530  -8.444425891
  H    0.880759810  -1.754030905  -8.587528398
  O    0.660410505  -3.370608496  -9.810367717
  C   -0.564161447  -4.044826497 -10.048848396
  H   -0.830169103  -3.825489457 -11.115085605
  C   -0.663380873  -3.131855236  -7.806129613
  H   -0.610248949  -3.901647826  -7.003456681
  C   -1.550962083  -3.610247710  -8.969962643
  H   -2.183127879  -2.761206346  -9.322660441
  H   -2.279800978  -4.381671557  -8.656894867
  O   -1.188800582  -1.884493695  -7.337192634
  N   -0.258727336  -5.529074915  -9.972753496
  C    0.171717291  -6.315497791 -11.068393598
  C    0.513517034  -7.596484455 -10.546326661
  N    0.298363507  -7.565944330  -9.157526098
  C   -0.148107500  -6.327554500  -8.822132271
  N    0.237154916  -5.949647087 -12.380031411
  C    0.703513692  -6.929338575 -13.230848985
  N    1.080653315  -8.218397881 -12.793612990
  C    1.006152886  -8.653519713 -11.400266554
  N    0.795448861  -6.612967770 -14.559177139
  H    0.535974184  -5.678195345 -14.872297992
  H    1.141489729  -7.255367160 -15.257914585
  O    1.348487045  -9.775614076 -11.138386170
  H   -0.424613624  -5.995117289  -7.779607582
  H    1.433513558  -8.920739336 -13.461755597
  P   -1.204061973  -1.727068550  -5.674515590
  O   -1.802042251  -0.184070230  -5.690992899
  C   -0.820162877   0.833156645  -5.374434414
  H   -0.847932600   1.529975020  -6.241254115
  H    0.213190083   0.429947747  -5.273603114
  C   -1.267567543   1.537034010  -4.088697559
  H   -0.448012008   1.596494032  -3.329557283
  O   -1.500023935   2.943673661  -4.429568670
  C   -2.833370056   3.328009645  -4.140164301
  H   -2.759995036   4.345623010  -3.693837651
  C   -2.589326777   1.001513395  -3.470971979
  H   -3.104514379   0.239061522  -4.107801905
  C   -3.457832799   2.252204026  -3.251685744
  H   -3.413097692   2.538205409  -2.171889933
  H   -4.529524416   2.063015721  -3.443151861
  O   -2.334016650   0.467778833  -2.168620905
  O    0.159385582  -1.911795838  -5.116356610
  O   -2.431017482  -2.760079484  -5.431027703
  N   -3.551168663   3.428796452  -5.461777188
  C   -4.032459204   4.579262079  -6.079082205
  C   -4.627618516   4.166847168  -7.316595622
  N   -4.476575718   2.781026890  -7.471187191
  C   -3.833602954   2.352574891  -6.385339168
  N   -4.032958905   5.923487800  -5.701473364
  C   -4.642097535   6.825332424  -6.599034126
  N   -5.216943661   6.481235392  -7.737583650
  C   -5.252334974   5.135537679  -8.163781177
  H   -4.636029221   7.900898658  -6.309355545
  N   -5.871214876   4.847896789  -9.319690515
  H   -6.306352113   5.573731579  -9.888850400
  H   -5.923333427   3.887980193  -9.661416238
  H   -3.521043104   1.323223501  -6.182341252
  P   -1.709582248  -1.076153750  -2.133905936
  O   -0.228070753  -0.580202593  -1.666990593
  C    0.841647286  -1.563956213  -1.649857274
  H    0.949621162  -2.049552866  -2.650786583
  H    1.737105402  -0.946721939  -1.434481069
  C    0.473879869  -2.523298280  -0.521256933
  H    0.366792808  -2.009303128   0.465300554
  O   -0.895728312  -2.915513403  -0.891114494
  C   -0.995355666  -4.359724903  -1.095262770
  H   -1.942293695  -4.628073921  -0.560583240
  C    1.299783723  -3.818874385  -0.379048474
  H    2.122258995  -3.904167973  -1.134752737
  C    0.271145427  -4.951225207  -0.483939348
  H    0.695069451  -5.790335919  -1.092719311
  H    0.007279876  -5.417874228   0.503170194
  O    1.961777426  -3.689812466   0.893395945
  O   -1.815187519  -1.744840292  -3.470166700
  O   -2.743288971  -1.597827951  -1.001366920
  N   -1.154645104  -4.622718169  -2.551010391
  C   -0.002650360  -4.647437928  -3.404634566
  N   -0.206669911  -5.137777446  -4.713932577
  C   -1.436490320  -5.650404192  -5.212015991
  C   -2.547643993  -5.652175305  -4.258992097
  C   -2.375722561  -5.192880379  -2.991588564
  O    1.093142729  -4.257556299  -3.044117102
  H    0.630633142  -5.118619116  -5.342116585
  O   -1.450316799  -6.007567601  -6.387587397
  C   -3.841436832  -6.222705451  -4.726039787
  H   -4.672553386  -5.510930888  -4.623990924
  H   -3.798356000  -6.513252765  -5.789654187
  H   -4.101750934  -7.134399260  -4.161569122
  H   -3.182321229  -5.239005053  -2.244633184
  P    2.196407597  -5.010846505   1.851446841
  O    3.290172669  -5.830813501   0.934202132
  C    3.551004561  -7.199240522   1.354222996
  H    4.516720037  -7.150593448   1.902837886
  H    2.758090430  -7.599959416   2.022054180
  C    3.673354120  -8.071971847   0.099738116
  H    4.562470266  -8.744495758   0.183779709
  O    2.565048380  -9.017028048   0.071091926
  C    1.640378725  -8.726611258  -0.990581228
  H    1.491173407  -9.699438644  -1.519145004
  C    3.643467237  -7.315065954  -1.252158389
  H    3.871412783  -6.225968089  -1.151940026
  C    2.254879085  -7.611790201  -1.844697472
  H    1.624106715  -6.693170396  -1.869044952
  H    2.322332462  -7.901825033  -2.912463851
  O    4.691230944  -7.761548316  -2.096655050
  O    1.029158989  -5.794652560   2.270663021
  O    3.061418215  -4.292658904   3.021119260
  N    0.334972392  -8.344746099  -0.339092323
  C   -0.749101443  -7.763913742  -1.148356092
  N   -1.906549451  -7.371293944  -0.499489925
  C   -2.071231065  -7.634468096   0.833716694
  C   -1.057027435  -8.289620788   1.610904790
  C    0.130051182  -8.614983721   1.009337617
  O   -0.523980112  -7.592719444  -2.335629161
  N   -3.235469846  -7.186122442   1.414635071
  H   -3.462874354  -7.434089926   2.366519203
  H   -3.994634347  -6.848125468   0.840618396
  H    1.460907895  -3.118298070  -5.830269434
  H    0.954531831  -9.098781377   1.559788220
  H   -1.208136312  -8.490001831   2.670123194
  H    4.632333971  -8.721860897  -2.295785974
  H   -2.611243136  -3.016869914  -4.451272475
  H   -2.378989527  -1.971417159  -0.136742076
  H    3.750835803  -3.618720840   2.756513725
  H    0.443269501  -8.362329831  -8.541757355
  H   -3.577206909   6.239399735  -4.842214411

