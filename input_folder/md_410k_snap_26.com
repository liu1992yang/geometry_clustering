%mem=64gb
%nproc=28       
%Chk=snap_26.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_26 

2     1 
  O    2.488286918  -4.259849696  -7.782999665
  C    2.455052009  -2.868257751  -8.144264385
  H    3.329954617  -2.772682037  -8.825739523
  H    2.628245224  -2.265741402  -7.225581029
  C    1.166529726  -2.433108653  -8.841179881
  H    1.193803133  -1.333190693  -9.047084114
  O    1.131182276  -3.042099613 -10.179517268
  C   -0.119768024  -3.665029288 -10.422113131
  H   -0.375375561  -3.469694878 -11.491831570
  C   -0.178202531  -2.821113502  -8.168346326
  H   -0.076666650  -3.684687223  -7.468300014
  C   -1.096898028  -3.179239027  -9.350010247
  H   -1.655415107  -2.276533681  -9.684369817
  H   -1.877950312  -3.912627896  -9.078696921
  O   -0.714934045  -1.655345775  -7.537274197
  N    0.101882747  -5.156953111 -10.272427495
  C   -0.717837739  -6.170444192 -10.810742567
  C   -0.199074827  -7.407547701 -10.338224306
  N    0.912273176  -7.120768393  -9.517988035
  C    1.077197458  -5.775443359  -9.479113016
  N   -1.807784187  -6.018024392 -11.624594260
  C   -2.407281548  -7.196041368 -12.000522413
  N   -1.947325375  -8.466952891 -11.592492831
  C   -0.798917535  -8.666995222 -10.708137000
  N   -3.538046221  -7.112784284 -12.776546482
  H   -3.845829246  -6.201566072 -13.109412879
  H   -3.959836641  -7.921886385 -13.209901551
  O   -0.512180430  -9.795686289 -10.408972202
  H    1.870931728  -5.157531320  -8.865890942
  H   -2.426021907  -9.325355089 -11.903019440
  P   -0.637020527  -1.676205887  -5.869344947
  O   -1.263235258  -0.151687113  -5.744644096
  C   -0.424533877   0.812986107  -5.066375624
  H   -0.192594580   1.585444120  -5.833471136
  H    0.526635095   0.372632128  -4.691754913
  C   -1.261264780   1.405375490  -3.928433469
  H   -0.669209029   1.526750883  -2.987281376
  O   -1.579494377   2.788185851  -4.295257803
  C   -2.978018982   3.024542418  -4.258658063
  H   -3.099029145   4.001965008  -3.738616712
  C   -2.619124355   0.690634545  -3.678499399
  H   -2.889089052  -0.051464930  -4.470129797
  C   -3.653642723   1.825668069  -3.592018987
  H   -3.884481228   2.026799168  -2.517823821
  H   -4.624145044   1.543420581  -4.040664481
  O   -2.614297009   0.059522173  -2.397422353
  O    0.729100501  -1.955655998  -5.388907927
  O   -1.857917475  -2.725459560  -5.638037739
  N   -3.420852849   3.175990029  -5.690851767
  C   -3.991182597   4.296163523  -6.288226177
  C   -4.169944743   3.979806329  -7.675810374
  N   -3.690243624   2.685145270  -7.926436555
  C   -3.242276087   2.219579870  -6.760626614
  N   -4.391214908   5.533570096  -5.779173146
  C   -4.958637001   6.435824169  -6.703070309
  N   -5.139719412   6.181632857  -7.986257651
  C   -4.756349785   4.946262071  -8.552964715
  H   -5.275679279   7.427525492  -6.306939120
  N   -4.968203182   4.763998973  -9.866130039
  H   -5.383846535   5.495312747 -10.444725435
  H   -4.708460207   3.891012625 -10.324213307
  H   -2.776875630   1.242373448  -6.584451825
  P   -1.761990593  -1.371480389  -2.285613491
  O   -0.768638198  -0.728900155  -1.169611132
  C    0.367773525  -1.488186352  -0.685505349
  H    1.033253462  -1.777334070  -1.531444732
  H    0.886583598  -0.751487417  -0.036472697
  C   -0.210574795  -2.670072432   0.087380179
  H   -0.766226305  -2.368048875   1.007964229
  O   -1.245860213  -3.163305714  -0.830954159
  C   -1.024930946  -4.566454839  -1.170797341
  H   -2.030201548  -5.030320256  -1.004398034
  C    0.751436086  -3.842404477   0.374586413
  H    1.783873366  -3.667046162  -0.019705763
  C    0.075419913  -5.071735645  -0.245974007
  H    0.839302221  -5.687328462  -0.786135255
  H   -0.366847206  -5.762762571   0.519336454
  O    0.842466783  -3.880585203   1.811613726
  O   -1.196941839  -1.830919546  -3.593483568
  O   -3.040400545  -2.272694570  -1.837905003
  N   -0.671658210  -4.645420946  -2.614025276
  C    0.622252596  -4.223570478  -3.068570982
  N    0.919917309  -4.480471727  -4.425916283
  C    0.061742531  -5.178586371  -5.314189279
  C   -1.222985949  -5.611687889  -4.767656018
  C   -1.537224811  -5.370223430  -3.469101836
  O    1.424877163  -3.677081119  -2.333740932
  H    1.780174514  -4.031147759  -4.793128378
  O    0.449606243  -5.317107934  -6.475918368
  C   -2.129795364  -6.342296638  -5.695289695
  H   -3.189378734  -6.114335502  -5.519910236
  H   -1.913181613  -6.096576364  -6.746932740
  H   -1.998018794  -7.433305118  -5.584717287
  H   -2.473214651  -5.732926208  -3.018837052
  P    1.360183999  -5.280633748   2.510592010
  O    2.798177944  -5.520375964   1.775294402
  C    3.478488149  -6.757538426   2.147572307
  H    4.365263464  -6.406986811   2.723110216
  H    2.858489962  -7.430094131   2.775269727
  C    3.904847946  -7.473604382   0.865315471
  H    4.829384081  -8.073727266   1.059340363
  O    2.918960601  -8.491044322   0.531331818
  C    2.341863365  -8.266728740  -0.767610310
  H    2.581241845  -9.184043005  -1.359984880
  C    4.064385097  -6.555474393  -0.373746390
  H    4.063294522  -5.469187941  -0.114162843
  C    2.928197494  -6.963380696  -1.322443917
  H    2.164381797  -6.155891018  -1.393449616
  H    3.282496660  -7.082016062  -2.366595088
  O    5.345510209  -6.723758563  -0.953078562
  O    0.413130682  -6.408216123   2.456649400
  O    1.791039767  -4.715850935   3.972367251
  N    0.848015234  -8.213576499  -0.588187521
  C    0.008571770  -7.903349476  -1.765531465
  N   -1.364219824  -7.922854379  -1.601291710
  C   -1.910511493  -8.290715388  -0.403171475
  C   -1.104867309  -8.568139530   0.747554952
  C    0.263633761  -8.521106884   0.633424876
  O    0.590409286  -7.588385532  -2.788296521
  N   -3.292135034  -8.291435534  -0.332182100
  H   -3.751889513  -8.733209738   0.452112980
  H   -3.829228484  -8.260958372  -1.190154443
  H    1.845627909  -4.439464707  -7.019305983
  H    0.934587656  -8.714559912   1.489368085
  H   -1.557891405  -8.779430345   1.715819114
  H    5.494964118  -7.637704887  -1.280867365
  H   -1.998859597  -3.023291363  -4.665390776
  H   -3.206001710  -2.422973548  -0.859323391
  H    1.111756386  -4.699538782   4.701256568
  H    1.461542956  -7.829678608  -9.042617267
  H   -4.247100704   5.783651260  -4.798062837

