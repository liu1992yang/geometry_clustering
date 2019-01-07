%mem=64gb
%nproc=28       
%Chk=snap_27.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_27 

2     1 
  O    2.421841501  -4.171697944  -7.745685222
  C    2.387227155  -2.778477214  -8.108058643
  H    3.278033350  -2.676348100  -8.767148277
  H    2.529154951  -2.178833803  -7.181160550
  C    1.116700855  -2.355943372  -8.843460033
  H    1.131736645  -1.254773962  -9.042877876
  O    1.142531674  -2.958109568 -10.185047098
  C   -0.095271171  -3.583437456 -10.484623650
  H   -0.302324964  -3.385049842 -11.564037253
  C   -0.246839458  -2.773190736  -8.226979580
  H   -0.163601984  -3.661522978  -7.555234397
  C   -1.117834320  -3.097836729  -9.453717987
  H   -1.645595748  -2.181526204  -9.800760558
  H   -1.922621927  -3.820705936  -9.228333481
  O   -0.816180312  -1.639317723  -7.567493725
  N    0.126644072  -5.073918650 -10.327008768
  C   -0.656825625  -6.089482329 -10.912399671
  C   -0.177323241  -7.324850752 -10.394270386
  N    0.878934092  -7.034569437  -9.504866489
  C    1.048880312  -5.690117743  -9.470202320
  N   -1.683535333  -5.937035156 -11.804425159
  C   -2.259479303  -7.115768978 -12.213251594
  N   -1.839262700  -8.384915880 -11.758866323
  C   -0.758098614  -8.584373832 -10.793104800
  N   -3.326584108  -7.037485163 -13.074323576
  H   -3.604346554  -6.131479350 -13.444826533
  H   -3.727881998  -7.852693760 -13.515018837
  O   -0.504792026  -9.713242025 -10.465055874
  H    1.819990571  -5.072060678  -8.827771235
  H   -2.302032137  -9.243434711 -12.092815405
  P   -0.698874704  -1.698154335  -5.901712012
  O   -1.269934477  -0.157887017  -5.726620365
  C   -0.375572285   0.760076419  -5.051864206
  H   -0.094002809   1.515472842  -5.819410921
  H    0.545695734   0.263902147  -4.670330692
  C   -1.181441290   1.401829111  -3.919465558
  H   -0.575944838   1.516188976  -2.986292809
  O   -1.455846649   2.789126269  -4.309695670
  C   -2.842712054   3.080699102  -4.236112047
  H   -2.910224655   4.082292390  -3.753596641
  C   -2.563236013   0.742409494  -3.644356283
  H   -2.891266915   0.034913733  -4.445916545
  C   -3.532277181   1.929119100  -3.502122151
  H   -3.670454487   2.156458553  -2.417407593
  H   -4.546717650   1.690777390  -3.870661724
  O   -2.562980533   0.078468041  -2.381201246
  O    0.675128713  -2.022428334  -5.468322936
  O   -1.930206203  -2.730742090  -5.658429028
  N   -3.334751693   3.188648247  -5.657174182
  C   -3.975234617   4.266104309  -6.262049241
  C   -4.229086409   3.884361554  -7.621409821
  N   -3.726243619   2.595108701  -7.849839539
  C   -3.194940777   2.192531874  -6.696197022
  N   -4.380537132   5.515478419  -5.786631560
  C   -5.031277128   6.357322510  -6.713187052
  N   -5.287426607   6.037475177  -7.968747323
  C   -4.904317106   4.787376622  -8.501148553
  H   -5.352714138   7.358195477  -6.343801190
  N   -5.203065326   4.526205817  -9.783855326
  H   -5.688244534   5.207892649 -10.367922057
  H   -4.942477169   3.638533661 -10.212729608
  H   -2.689539474   1.236243145  -6.512515508
  P   -1.741550486  -1.371202638  -2.305888837
  O   -0.689871970  -0.750502370  -1.229488937
  C    0.445302483  -1.538972373  -0.794852457
  H    1.067611705  -1.841969634  -1.668842967
  H    1.007586993  -0.818497387  -0.164583187
  C   -0.135189116  -2.710016214  -0.007057197
  H   -0.670612617  -2.392604175   0.920344335
  O   -1.194746596  -3.192312816  -0.902556391
  C   -0.992819569  -4.596734932  -1.253698054
  H   -1.995378514  -5.055022212  -1.057632776
  C    0.814544469  -3.894452025   0.273376411
  H    1.850148593  -3.725025405  -0.112988113
  C    0.131406789  -5.108762838  -0.362344789
  H    0.886323835  -5.708125075  -0.934701933
  H   -0.293366629  -5.826845424   0.388605449
  O    0.916332250  -3.933862408   1.708792187
  O   -1.227231779  -1.840921619  -3.630643136
  O   -3.018935942  -2.242727829  -1.807988349
  N   -0.688502307  -4.673983108  -2.707336367
  C    0.601134384  -4.274155866  -3.195225031
  N    0.853562920  -4.540226195  -4.559298481
  C   -0.036821208  -5.229309136  -5.424365538
  C   -1.297757790  -5.671465845  -4.833447151
  C   -1.570852897  -5.418752870  -3.526925453
  O    1.429923321  -3.734480689  -2.485571416
  H    1.708018276  -4.104464821  -4.955192354
  O    0.320899463  -5.348581304  -6.598012971
  C   -2.234597554  -6.423182362  -5.713701312
  H   -3.275490264  -6.089250383  -5.602255859
  H   -1.972219140  -6.309721780  -6.777540979
  H   -2.200536355  -7.503216737  -5.486172001
  H   -2.488758788  -5.785434474  -3.042928337
  P    1.263698401  -5.354747470   2.475368052
  O    2.827072912  -5.544371613   2.012427858
  C    3.432099488  -6.827653691   2.335251416
  H    4.306451633  -6.539697114   2.962348692
  H    2.760552858  -7.495321576   2.914441137
  C    3.892247272  -7.519327892   1.048923397
  H    4.825104022  -8.104211771   1.253031126
  O    2.930851453  -8.552643239   0.694654369
  C    2.380561236  -8.346098251  -0.618476981
  H    2.627824814  -9.273652805  -1.191932599
  C    4.055275221  -6.591523004  -0.182309680
  H    3.979809878  -5.508539280   0.080259606
  C    2.992694557  -7.061017150  -1.186681074
  H    2.230752065  -6.269942709  -1.370591624
  H    3.424931853  -7.228497930  -2.194858942
  O    5.369993421  -6.675293069  -0.702940027
  O    0.383162395  -6.507485578   2.251482452
  O    1.336165951  -4.783054592   3.992156853
  N    0.883754250  -8.289668815  -0.465754792
  C    0.059098520  -7.926513383  -1.635431009
  N   -1.314797160  -7.940815091  -1.488410036
  C   -1.877433075  -8.353588064  -0.310923072
  C   -1.087760030  -8.700579911   0.830097017
  C    0.281907307  -8.648343252   0.735199228
  O    0.650643656  -7.566525220  -2.639455046
  N   -3.258417909  -8.337689732  -0.255418139
  H   -3.735513887  -8.793758718   0.509371205
  H   -3.784578516  -8.255307190  -1.115454295
  H    1.733404984  -4.359474887  -7.029156183
  H    0.940953797  -8.887971964   1.587617756
  H   -1.553746893  -8.956161921   1.780614205
  H    5.591168562  -7.572963670  -1.034998979
  H   -2.058266316  -3.034818221  -4.685110575
  H   -3.100933298  -2.469821038  -0.832082306
  H    1.819266616  -3.923278098   4.153959251
  H    1.385847388  -7.738518382  -8.977029508
  H   -4.184976016   5.816337298  -4.829392055
