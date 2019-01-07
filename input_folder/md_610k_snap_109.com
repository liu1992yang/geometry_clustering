%mem=64gb
%nproc=28       
%Chk=snap_109.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_109 

2     1 
  O    3.465449026  -4.321491988  -6.299601911
  C    3.517494703  -3.106353958  -7.041708195
  H    4.162563402  -3.256853617  -7.930767837
  H    4.038984834  -2.414653482  -6.343113144
  C    2.148001848  -2.560467880  -7.450475137
  H    2.179807777  -1.460799150  -7.654661752
  O    1.811383418  -3.129136673  -8.762072634
  C    0.668692126  -3.962790253  -8.683268410
  H    0.067026707  -3.766209796  -9.605188258
  C    0.984361143  -2.918479363  -6.499732091
  H    1.332988914  -3.477699005  -5.586294334
  C   -0.014893113  -3.728425405  -7.339562141
  H   -0.979732508  -3.187364991  -7.463250697
  H   -0.291564789  -4.678342704  -6.824854221
  O    0.417878913  -1.642378201  -6.133147828
  N    1.192120944  -5.386008688  -8.793515140
  C    0.470161989  -6.570533406  -8.539102669
  C    1.380909752  -7.651263446  -8.734170390
  N    2.632397031  -7.109762828  -9.050635971
  C    2.511079922  -5.753138397  -9.095907581
  N   -0.860986211  -6.697994498  -8.247526212
  C   -1.292589254  -7.984651590  -8.086049028
  N   -0.475110832  -9.113100737  -8.301055596
  C    0.955126829  -9.027584873  -8.568653116
  N   -2.596262639  -8.190030358  -7.638279399
  H   -3.143192409  -7.320460251  -7.470821095
  H   -3.146149160  -8.937080819  -8.052625151
  O    1.587692549 -10.047422455  -8.645408460
  H    3.293226175  -5.029176647  -9.347426401
  H   -0.842772284 -10.062422751  -8.148802688
  P   -0.439076278  -1.652062889  -4.713652726
  O   -0.664714270  -0.009291813  -4.683911947
  C    0.137161595   0.725153629  -3.726166173
  H    0.768944116   1.405544824  -4.344578987
  H    0.794906149   0.058872490  -3.126038853
  C   -0.809357920   1.526216419  -2.833851764
  H   -0.397338437   1.624604066  -1.796432715
  O   -0.824745471   2.887577903  -3.370888391
  C   -2.135168849   3.456937210  -3.281457073
  H   -2.021722819   4.397970553  -2.684712675
  C   -2.296577744   1.080326730  -2.794951162
  H   -2.605113960   0.491580346  -3.692782721
  C   -3.071444220   2.401801663  -2.683950115
  H   -3.292229785   2.611679414  -1.608480006
  H   -4.068723866   2.359732783  -3.158311716
  O   -2.531643795   0.352326431  -1.586872174
  O    0.334040377  -2.258764948  -3.610046204
  O   -1.821891426  -2.207252618  -5.364249541
  N   -2.533859886   3.851239056  -4.669062967
  C   -2.550782057   3.106969825  -5.845121893
  C   -3.064594989   3.968443399  -6.871838047
  N   -3.345884007   5.231269324  -6.330319479
  C   -3.027794625   5.162558927  -5.040294633
  N   -2.200579358   1.791853306  -6.137413354
  C   -2.358238016   1.378956042  -7.469435924
  N   -2.839889674   2.134990152  -8.445897040
  C   -3.210335747   3.475550870  -8.207458269
  H   -2.065485843   0.329315681  -7.700937725
  N   -3.678283778   4.201454794  -9.236692480
  H   -3.763473923   3.807821293 -10.172998854
  H   -3.953114690   5.176569579  -9.109865319
  H   -3.113120891   5.965887589  -4.311213926
  P   -2.381008558  -1.299039966  -1.723388903
  O   -1.202898765  -1.447654917  -0.618089384
  C   -0.438879216  -2.688355613  -0.667485969
  H   -0.035212218  -2.854922931  -1.695043027
  H    0.403809699  -2.491790277   0.036463975
  C   -1.380830501  -3.777142932  -0.175352991
  H   -1.680043055  -3.637105923   0.892069836
  O   -2.608429288  -3.515650091  -0.943181905
  C   -3.125536666  -4.743334600  -1.539027352
  H   -4.199008993  -4.741749463  -1.220682734
  C   -0.988608959  -5.241971923  -0.469195605
  H   -0.143028940  -5.340387650  -1.210306712
  C   -2.291481086  -5.889129531  -0.969685737
  H   -2.090538268  -6.711975382  -1.684059926
  H   -2.807977436  -6.386669341  -0.113648528
  O   -0.704915802  -5.915905224   0.770724024
  O   -2.095471613  -1.657161939  -3.156230342
  O   -3.883947399  -1.666075246  -1.229104447
  N   -3.062875322  -4.621007404  -3.020488262
  C   -1.978880361  -5.175979844  -3.782997671
  N   -2.269158958  -5.456249142  -5.138475132
  C   -3.495009600  -5.106638647  -5.796266180
  C   -4.375843212  -4.217211866  -5.015152304
  C   -4.177683025  -4.047075379  -3.685884017
  O   -0.893432013  -5.427619326  -3.297340247
  H   -1.582113754  -6.043735883  -5.657150663
  O   -3.702083554  -5.575301933  -6.896713227
  C   -5.476371453  -3.561533334  -5.769820301
  H   -5.111621575  -2.682386450  -6.324707078
  H   -5.909916659  -4.246299372  -6.523040723
  H   -6.308197088  -3.233612332  -5.131480421
  H   -4.855975974  -3.450698012  -3.056177165
  P    0.854224929  -5.892222116   1.290364754
  O    1.575699206  -6.192168288  -0.159445514
  C    3.015696755  -6.015454798  -0.221887238
  H    3.409436164  -5.435893601   0.636343415
  H    3.446385372  -7.042999343  -0.208372667
  C    3.335294018  -5.306691087  -1.537062975
  H    4.296787281  -4.740736192  -1.456837809
  O    3.601712435  -6.354842883  -2.534270762
  C    2.968114295  -6.047631313  -3.779584894
  H    3.730992693  -6.252414695  -4.566659313
  C    2.226519445  -4.428080343  -2.162473023
  H    1.198656637  -4.740783012  -1.845998662
  C    2.445351818  -4.616407052  -3.670915655
  H    1.527685833  -4.421671446  -4.265164429
  H    3.206081334  -3.901734301  -4.060672263
  O    2.458731983  -3.086982284  -1.744945991
  O    1.184528057  -6.688245391   2.459820981
  O    1.104551349  -4.260374070   1.417728810
  N    1.818114124  -7.019579437  -3.971853555
  C    1.051869419  -6.941402925  -5.205375748
  N   -0.026992970  -7.783976508  -5.378639247
  C   -0.397673772  -8.634554371  -4.367106252
  C    0.317020645  -8.690680762  -3.128538434
  C    1.403891977  -7.865436107  -2.954824691
  O    1.388388235  -6.121905982  -6.061089505
  N   -1.534745510  -9.374484596  -4.600661406
  H   -1.785725399 -10.147569641  -4.002788529
  H   -1.952166792  -9.349931645  -5.521215237
  H    2.796500148  -4.960972337  -6.648380481
  H    1.987873207  -7.848014302  -2.015753578
  H    0.010859193  -9.368683335  -2.333995087
  H    1.902339540  -2.474999704  -2.301610311
  H   -2.547974409  -2.474604586  -4.683852529
  H   -3.990938989  -2.076299966  -0.314223684
  H    1.399776538  -3.898780994   2.293834936
  H    3.466604583  -7.663986927  -9.229128549
  H   -1.732349480   1.179070396  -5.432686415

