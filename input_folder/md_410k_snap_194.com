%mem=64gb
%nproc=28       
%Chk=snap_194.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_194 

2     1 
  O    0.465927510  -1.570113635  -5.981988625
  C    0.692855250  -2.523206115  -7.013013724
  H    1.039443962  -3.489625813  -6.577121421
  H    1.528987159  -2.088228033  -7.598903178
  C   -0.567906936  -2.685077604  -7.876486914
  H   -0.814928153  -1.749428539  -8.430239074
  O   -0.253485947  -3.658142650  -8.926144949
  C   -1.216205727  -4.708706599  -8.966713739
  H   -1.417113339  -4.918188597 -10.046208724
  C   -1.817224932  -3.235107493  -7.142045985
  H   -1.578565582  -3.664417551  -6.140256739
  C   -2.403203037  -4.284716231  -8.098388845
  H   -3.211641588  -3.822930250  -8.715276912
  H   -2.901127382  -5.124185147  -7.578717865
  O   -2.767605530  -2.147581908  -7.072199339
  N   -0.528331318  -5.917293530  -8.376645496
  C   -0.917478413  -7.269871037  -8.479309163
  C   -0.005184746  -8.009526364  -7.667109480
  N    0.921831853  -7.115901059  -7.117599367
  C    0.595053218  -5.871366793  -7.524491676
  N   -1.958404820  -7.808846769  -9.173399136
  C   -2.071501120  -9.180782022  -9.059344536
  N   -1.184475737  -9.987876593  -8.303521122
  C   -0.120600830  -9.435534477  -7.490666449
  N   -3.069622950  -9.785404515  -9.774595067
  H   -3.710961676  -9.214012912 -10.322059359
  H   -3.303352460 -10.762270561  -9.669503750
  O    0.520378177 -10.162076219  -6.766185134
  H    1.119143210  -4.938358856  -7.228996032
  H   -1.369491624 -10.994677053  -8.182976890
  P   -3.042343073  -1.555124240  -5.560210166
  O   -3.078498529   0.077451806  -5.763050599
  C   -1.813635910   0.742660012  -5.949785770
  H   -2.144167158   1.770761858  -6.214210446
  H   -1.259312114   0.309450115  -6.808091798
  C   -0.958445088   0.732419154  -4.670803170
  H   -0.265288700  -0.154078659  -4.622473166
  O   -0.021720130   1.849706346  -4.756348366
  C   -0.263591300   2.819748134  -3.733905791
  H    0.751705710   3.093736764  -3.342216418
  C   -1.762463252   0.924611337  -3.362937632
  H   -2.869227713   0.951964178  -3.525254439
  C   -1.223481885   2.212418740  -2.717109202
  H   -0.681867283   1.943498043  -1.772532918
  H   -2.038448229   2.880097004  -2.385620395
  O   -1.387965053  -0.124647664  -2.444657741
  O   -1.873170567  -2.004072579  -4.684344721
  O   -4.571762256  -2.007320976  -5.389854529
  N   -0.821664905   4.013426440  -4.459978129
  C   -0.458338280   4.382392005  -5.752820167
  C   -1.104200340   5.629511728  -6.029301635
  N   -1.861107657   6.017058873  -4.914849301
  C   -1.691519465   5.067653095  -3.993619857
  N    0.358556818   3.757610322  -6.694425560
  C    0.532845440   4.441979707  -7.910731897
  N   -0.033696200   5.600470598  -8.207318734
  C   -0.880849065   6.266199187  -7.293630471
  H    1.193578190   3.965904716  -8.669732522
  N   -1.413306219   7.440177629  -7.663479119
  H   -1.210617727   7.854226249  -8.574947518
  H   -2.031284609   7.960041064  -7.038292206
  H   -2.129468946   5.057668889  -2.998915444
  P   -2.318783480  -1.490061443  -2.496879936
  O   -1.100900758  -2.555259053  -2.525934685
  C   -1.440892221  -3.959247654  -2.677306234
  H   -2.302296271  -4.124342131  -3.350725963
  H   -0.517294325  -4.343083642  -3.185553241
  C   -1.675780136  -4.563031770  -1.289725973
  H   -1.919937097  -3.788937898  -0.512853975
  O   -2.882380931  -5.370490970  -1.393102820
  C   -2.592079033  -6.760202944  -1.210933631
  H   -3.445326646  -7.146467094  -0.600660494
  C   -0.547462690  -5.500155449  -0.787390836
  H    0.315091809  -5.575893531  -1.491468983
  C   -1.221120118  -6.861564682  -0.539220288
  H   -0.602646453  -7.703537476  -0.906566012
  H   -1.332832840  -7.037321415   0.555077402
  O   -0.166437400  -4.945107392   0.489878503
  O   -3.563484970  -1.320664937  -3.338091031
  O   -2.766285760  -1.692450318  -0.939825907
  N   -2.646028167  -7.418998293  -2.566883681
  C   -1.594025664  -7.260667568  -3.512357433
  N   -1.760963098  -7.914723858  -4.755687465
  C   -2.924884957  -8.667160052  -5.144403014
  C   -3.986964446  -8.710522150  -4.131461387
  C   -3.828183617  -8.110031229  -2.925490900
  O   -0.589953666  -6.600040968  -3.297248814
  H   -0.956133966  -7.885670888  -5.393025005
  O   -2.908159154  -9.159778499  -6.254650324
  C   -5.220823585  -9.455541183  -4.503705155
  H   -5.631437450  -9.099016378  -5.463132683
  H   -5.012414003 -10.532197831  -4.629740201
  H   -6.025198351  -9.370084785  -3.759737346
  H   -4.614662749  -8.138371945  -2.156933151
  P    1.441713390  -4.945458072   0.845031470
  O    1.701638372  -3.332364504   0.670542199
  C    3.034534855  -2.897857139   0.314907665
  H    3.192179721  -2.037350536   1.005587689
  H    3.810803138  -3.669559219   0.501967009
  C    3.072025123  -2.436794727  -1.144175749
  H    3.739309491  -1.543755139  -1.258364306
  O    3.741443565  -3.488381496  -1.910213750
  C    3.077170106  -3.718895567  -3.157622244
  H    3.817673567  -3.456577587  -3.953695146
  C    1.710226799  -2.175621114  -1.821593535
  H    0.836914793  -2.486377228  -1.202933391
  C    1.787126026  -2.897577791  -3.175926034
  H    0.887895630  -3.521027380  -3.368334688
  H    1.800440333  -2.164448641  -4.012930722
  O    1.645044414  -0.754199993  -2.018446687
  O    2.301006241  -5.866865443   0.091282808
  O    1.387872855  -5.130858086   2.456573979
  N    2.813431421  -5.205272500  -3.262876868
  C    2.202882479  -5.673437264  -4.488547514
  N    2.126017960  -7.033768645  -4.724374941
  C    2.511816542  -7.933006161  -3.755534587
  C    3.085535801  -7.472553523  -2.526730155
  C    3.251162928  -6.124559262  -2.311920753
  O    1.784400054  -4.843568295  -5.295492646
  N    2.295468458  -9.253236266  -4.046053737
  H    2.633320242  -9.988529163  -3.442032302
  H    1.913494985  -9.537458095  -4.945697510
  H   -0.166834584  -1.932075540  -5.280285737
  H    3.730130589  -5.715367735  -1.397744408
  H    3.385886023  -8.175864156  -1.748259812
  H    0.723242960  -0.515823209  -2.272068767
  H   -4.999798735  -1.815835676  -4.468603952
  H   -3.673258364  -1.396213252  -0.656894312
  H    0.804108741  -4.510993119   2.983520770
  H    1.632212923  -7.357517555  -6.344493647
  H    0.786509358   2.844326554  -6.470936510

