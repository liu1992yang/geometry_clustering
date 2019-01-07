%mem=64gb
%nproc=28       
%Chk=snap_10.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_10 

2     1 
  O   -0.314413611  -6.244303092  -7.734075812
  C    0.752037094  -5.340840145  -7.429053359
  H    1.632430375  -5.776366107  -7.950696238
  H    0.947273419  -5.343809002  -6.337913918
  C    0.483823434  -3.918479654  -7.936498424
  H    1.057934225  -3.174920943  -7.324162189
  O    1.115596144  -3.772448319  -9.248146654
  C    0.156350816  -3.648451158 -10.277063943
  H    0.571666505  -2.896084386 -10.997691833
  C   -1.011033111  -3.550407402  -8.124390179
  H   -1.719162983  -4.313305758  -7.712163743
  C   -1.191894078  -3.316249227  -9.634311022
  H   -1.457395151  -2.242170966  -9.800649478
  H   -2.046599062  -3.879412132 -10.049287566
  O   -1.328930608  -2.258666328  -7.548788673
  N    0.122296128  -4.975416815 -11.004646374
  C    0.681724203  -5.221047912 -12.279658226
  C    0.546462729  -6.618196825 -12.522187002
  N   -0.051976946  -7.197411487 -11.388969880
  C   -0.285814335  -6.213431101 -10.482595852
  N    1.244786008  -4.314953505 -13.128653571
  C    1.738778691  -4.852274780 -14.299404277
  N    1.637316033  -6.221970026 -14.624140354
  C    1.013610548  -7.217429168 -13.752370343
  N    2.348098123  -3.991521702 -15.171505664
  H    2.441444099  -3.007683455 -14.923528459
  H    2.763771696  -4.287671976 -16.043319670
  O    0.967379400  -8.354593080 -14.136798604
  H   -0.693128862  -6.332347819  -9.437215007
  H    2.013300428  -6.583261752 -15.514922892
  P   -1.247340020  -2.172201233  -5.908918251
  O   -1.254256371  -0.524721731  -5.791467073
  C   -0.218642386   0.091733762  -4.992140411
  H    0.366120058   0.710661179  -5.711037712
  H    0.451982600  -0.655635666  -4.513389830
  C   -0.903157611   0.984223369  -3.955768194
  H   -0.252367915   1.136362956  -3.055730639
  O   -0.997895681   2.333349156  -4.527700806
  C   -2.327147017   2.825000521  -4.485820316
  H   -2.244789000   3.868883957  -4.104834746
  C   -2.343275772   0.573102365  -3.548039271
  H   -2.769702839  -0.241114952  -4.179592591
  C   -3.172208323   1.867131843  -3.645560360
  H   -3.338169227   2.264041732  -2.613344279
  H   -4.186277498   1.680925453  -4.043679262
  O   -2.342490665   0.231433546  -2.161699965
  O   -0.109182618  -2.909157373  -5.325826505
  O   -2.762473153  -2.512146478  -5.467724308
  N   -2.822177131   2.873261668  -5.909879963
  C   -3.398108539   3.949113896  -6.577818078
  C   -3.731591097   3.486329170  -7.894551706
  N   -3.340474011   2.146405611  -8.034749625
  C   -2.805294282   1.791993125  -6.866677745
  N   -3.681364673   5.260865381  -6.190642722
  C   -4.301111247   6.078170714  -7.157772918
  N   -4.635794965   5.683510543  -8.373790685
  C   -4.375371701   4.368914201  -8.816014542
  H   -4.523651364   7.128524139  -6.861633019
  N   -4.751961692   4.033162841 -10.061379075
  H   -5.215458527   4.702789469 -10.674978249
  H   -4.583347229   3.096540023 -10.426043787
  H   -2.377330665   0.811995742  -6.611550052
  P   -2.152579950  -1.376350156  -1.776710209
  O   -0.717407151  -1.159634996  -1.032210323
  C    0.076565739  -2.336919404  -0.737785409
  H    0.381034990  -2.836741324  -1.684535989
  H    0.972938566  -1.906553727  -0.236206715
  C   -0.736092776  -3.223695303   0.199885966
  H   -0.892828827  -2.769927474   1.206346547
  O   -2.078702534  -3.243448505  -0.406682487
  C   -2.510266609  -4.618567755  -0.676279484
  H   -3.573616369  -4.618209699  -0.330981744
  C   -0.279654688  -4.698429264   0.296137187
  H    0.500588837  -4.972199528  -0.462638556
  C   -1.569307329  -5.506125861   0.119198532
  H   -1.393539176  -6.515856230  -0.337108443
  H   -1.981946941  -5.784407115   1.122484599
  O    0.184303004  -4.969333727   1.635729024
  O   -2.202177733  -2.230943438  -2.995494645
  O   -3.419980163  -1.407595986  -0.762289446
  N   -2.475231044  -4.861023629  -2.150484070
  C   -1.241904863  -4.955292774  -2.872276242
  N   -1.347242541  -5.239244076  -4.258851994
  C   -2.569472793  -5.351418057  -4.979639874
  C   -3.780620657  -5.137771949  -4.184391810
  C   -3.702594209  -4.881738328  -2.851374987
  O   -0.145017734  -4.848542916  -2.358383331
  H   -0.450428479  -5.206299480  -4.778733078
  O   -2.485668737  -5.594692578  -6.179763388
  C   -5.081720938  -5.187188389  -4.906681611
  H   -4.944172838  -5.415538082  -5.978263689
  H   -5.745342944  -5.967528776  -4.503607257
  H   -5.616784819  -4.227439703  -4.855844436
  H   -4.604460581  -4.693907524  -2.249920657
  P    1.791237687  -4.772726636   1.889570353
  O    2.405670455  -5.060577762   0.393442727
  C    3.718518462  -5.682983706   0.330631625
  H    4.396629764  -4.864891976   0.007935097
  H    4.045594986  -6.088754579   1.313449627
  C    3.658101617  -6.803984751  -0.713329086
  H    4.656656410  -6.932687159  -1.196657543
  O    3.457885350  -8.074672414  -0.043422067
  C    2.093547774  -8.528860776  -0.180343595
  H    2.190846847  -9.624065743  -0.379197257
  C    2.514266329  -6.679476307  -1.754712781
  H    2.105223037  -5.645402565  -1.853317339
  C    1.470755336  -7.720670909  -1.323501481
  H    0.515503549  -7.231222722  -1.033042967
  H    1.170560834  -8.375743740  -2.162941744
  O    3.012189202  -6.928303929  -3.064917216
  O    2.386936891  -5.425612895   3.052282345
  O    1.849764000  -3.117818376   1.858759365
  N    1.446458412  -8.340982209   1.161792719
  C   -0.014567465  -8.205980286   1.317521136
  N   -0.515592781  -8.038325942   2.588288685
  C    0.307273830  -8.098906425   3.677096681
  C    1.722475435  -8.350187417   3.548756212
  C    2.256322528  -8.446146992   2.295091703
  O   -0.684291215  -8.217770416   0.293979911
  N   -0.273140333  -7.909748256   4.904448061
  H    0.264349330  -7.943785911   5.753940589
  H   -1.266955876  -7.745416088   4.985052584
  H   -1.041533300  -6.191794592  -7.030414514
  H    3.335732679  -8.596177842   2.131579425
  H    2.355764575  -8.417597343   4.430525942
  H    3.369595225  -7.838280003  -3.155837103
  H   -2.925997636  -2.624653186  -4.439564908
  H   -3.286251537  -1.783812017   0.163025896
  H    2.276592917  -2.653698872   2.622242109
  H   -0.251898466  -8.190414441 -11.298066034
  H   -3.422661355   5.615350029  -5.267225247
