%mem=64gb
%nproc=28       
%Chk=snap_47.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_47 

2     1 
  O    2.146108890  -4.521616160  -5.882288545
  C    2.787054448  -3.586228671  -6.762524722
  H    3.773311996  -4.070379315  -6.947719761
  H    2.949083633  -2.633964756  -6.216040016
  C    2.040696707  -3.362493475  -8.078435483
  H    2.434533400  -2.459519689  -8.608993639
  O    2.365524525  -4.470723889  -8.986160503
  C    1.185777065  -5.032764690  -9.546421566
  H    1.417403618  -5.268361741 -10.613655700
  C    0.492535432  -3.350521052  -7.988704832
  H    0.105849643  -3.901098734  -7.087793948
  C    0.033939109  -4.058956147  -9.269095073
  H   -0.107609212  -3.331933790 -10.096813154
  H   -0.949997573  -4.554115358  -9.139346688
  O   -0.012946439  -1.999510619  -8.014703440
  N    0.951528935  -6.341836683  -8.825579416
  C    0.253344029  -7.451327152  -9.337392304
  C    0.145154169  -8.393882167  -8.273926926
  N    0.755165572  -7.834570849  -7.138892013
  C    1.229198651  -6.613307748  -7.469491191
  N   -0.237001191  -7.637354706 -10.604304338
  C   -0.866319258  -8.838493932 -10.802592682
  N   -0.996222502  -9.835522638  -9.800353494
  C   -0.502676932  -9.665808844  -8.444750372
  N   -1.436248351  -9.062629569 -12.032234005
  H   -1.302748159  -8.375252294 -12.772333591
  H   -1.797815103  -9.963351916 -12.310409163
  O   -0.680858136 -10.565133948  -7.653325636
  H    1.783327565  -5.898911957  -6.769286025
  H   -1.477306710 -10.723725845 -10.001328672
  P    0.021978570  -1.268364565  -6.538688662
  O   -1.428818495  -0.458283881  -6.508418914
  C   -1.313350629   0.977559933  -6.430443750
  H   -2.330501430   1.318530855  -6.712449318
  H   -0.583178325   1.379213070  -7.162821095
  C   -0.944510117   1.364077982  -4.988283677
  H    0.144714098   1.206773152  -4.748938986
  O   -1.109471354   2.808039476  -4.850585443
  C   -2.177880423   3.125674053  -3.958981804
  H   -1.775129702   3.935524690  -3.295657120
  C   -1.872737277   0.695217548  -3.944699611
  H   -2.565107456  -0.052116728  -4.400157099
  C   -2.612987336   1.851663074  -3.243107933
  H   -2.305870282   1.887844623  -2.168778425
  H   -3.701796444   1.675089486  -3.216390910
  O   -0.967172457   0.082616076  -3.014760157
  O    1.173760797  -0.400655442  -6.257384807
  O   -0.240873369  -2.526403344  -5.568249174
  N   -3.243983853   3.735809367  -4.829525497
  C   -2.992349166   4.655384648  -5.845119941
  C   -4.263360373   5.111354467  -6.321340651
  N   -5.284674890   4.463053898  -5.613048583
  C   -4.681638432   3.658091718  -4.736690459
  N   -1.797998862   5.134380738  -6.381568709
  C   -1.920701649   6.107890077  -7.390517900
  N   -3.071844928   6.573999343  -7.847746431
  C   -4.308162859   6.112993339  -7.345237859
  H   -0.977355788   6.514009482  -7.821863566
  N   -5.435074022   6.642347268  -7.844808136
  H   -5.412239802   7.360429530  -8.570627483
  H   -6.351541320   6.350273488  -7.502399437
  H   -5.175909749   3.011858039  -4.017473843
  P   -1.387262206  -1.343844375  -2.297697492
  O   -0.905425691  -0.995415662  -0.792986468
  C   -0.410875568  -2.137823158  -0.033991367
  H    0.457288918  -2.596173838  -0.549783209
  H   -0.048105481  -1.660961469   0.907239836
  C   -1.590073439  -3.078859764   0.209834985
  H   -2.241721211  -2.745345180   1.049942148
  O   -2.427268197  -2.963552806  -0.997699599
  C   -2.934821836  -4.292790644  -1.407660620
  H   -4.042699133  -4.141699605  -1.417359753
  C   -1.239689903  -4.584244846   0.305954526
  H   -0.275536108  -4.816760448  -0.226955831
  C   -2.443702148  -5.279730527  -0.355132899
  H   -2.174435239  -6.274311697  -0.764962392
  H   -3.219840524  -5.494469269   0.415247688
  O   -1.232980607  -5.033768827   1.655581728
  O   -0.753370067  -2.464970731  -3.056988631
  O   -3.021396379  -1.174544617  -2.393694375
  N   -2.465541828  -4.567648238  -2.783013832
  C   -1.190954348  -5.180957230  -3.039214288
  N   -0.877877057  -5.412984210  -4.392857690
  C   -1.651520511  -4.952440027  -5.501907298
  C   -2.822068122  -4.132242332  -5.149003450
  C   -3.182700562  -3.965457697  -3.854335039
  O   -0.457091602  -5.581976436  -2.149724213
  H   -0.001661663  -5.956912614  -4.586052268
  O   -1.269366904  -5.282002236  -6.611751123
  C   -3.545098865  -3.480569308  -6.274622345
  H   -4.630572238  -3.650165934  -6.232842209
  H   -3.367897765  -2.392701562  -6.287898642
  H   -3.203986858  -3.864194598  -7.252639318
  H   -4.056658971  -3.363072073  -3.572655217
  P    0.169816845  -4.919926413   2.538107521
  O    1.065795498  -6.258409991   2.225111829
  C    1.186290716  -6.891488113   0.944191919
  H    1.338511095  -7.961680145   1.224962214
  H    0.273428116  -6.822207451   0.330804436
  C    2.452831788  -6.340240640   0.273329303
  H    3.215500433  -6.091243415   1.053369212
  O    3.073259465  -7.421558053  -0.474376585
  C    3.147483827  -7.120667546  -1.876353840
  H    4.177119788  -7.435494330  -2.182024065
  C    2.218732450  -5.181877282  -0.726977993
  H    1.130895453  -4.912784792  -0.853569205
  C    2.847678961  -5.632301426  -2.045886485
  H    2.171204218  -5.409637873  -2.911360660
  H    3.763588790  -5.061820972  -2.296366463
  O    2.769165338  -3.973350590  -0.201469263
  O   -0.039519440  -4.693607955   3.952851710
  O    0.866986628  -3.741047078   1.621683696
  N    2.164938687  -8.059330551  -2.545863473
  C    1.582756506  -7.776667726  -3.839153351
  N    0.862295014  -8.771861934  -4.478226520
  C    0.663191279  -9.998332931  -3.894941759
  C    1.170110505 -10.256848177  -2.569988614
  C    1.920312633  -9.297626371  -1.945475990
  O    1.727581295  -6.679298039  -4.373743548
  N   -0.015623632 -10.922979323  -4.629903946
  H   -0.182294004 -11.858483751  -4.289519052
  H   -0.312330531 -10.721615783  -5.588380631
  H    1.249180685  -4.208598502  -5.599792473
  H    2.368670886  -9.459306969  -0.947190544
  H    0.975904223 -11.211866079  -2.082210794
  H    3.755308818  -3.974392530  -0.205108453
  H   -0.444717058  -2.354241078  -4.516046864
  H   -3.554929722  -1.453133767  -1.584684960
  H    1.836765008  -3.772412261   1.358532686
  H    0.774206204  -8.294830292  -6.186086551
  H   -0.899708148   4.739384961  -6.064415512

