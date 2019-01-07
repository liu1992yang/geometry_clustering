%mem=64gb
%nproc=28       
%Chk=snap_135.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_135 

2     1 
  O    3.013009469  -3.717558107  -6.358245885
  C    2.984515438  -2.921157339  -7.566580495
  H    3.539723326  -3.433102228  -8.378849536
  H    3.467050921  -1.950963591  -7.332706514
  C    1.507673363  -2.741770091  -7.933313760
  H    1.339749835  -1.788981962  -8.495643862
  O    1.141213584  -3.777934711  -8.899133812
  C    0.096684000  -4.610638452  -8.412691687
  H   -0.584003642  -4.791199872  -9.283241755
  C    0.528451498  -2.876980161  -6.741890437
  H    1.053007652  -3.157134615  -5.792053214
  C   -0.513168965  -3.922680563  -7.186406240
  H   -1.469278120  -3.423834476  -7.467466542
  H   -0.780577386  -4.626436537  -6.374285638
  O   -0.062521048  -1.565771082  -6.611241888
  N    0.724882132  -5.931000651  -8.045563102
  C    0.349504847  -7.219675476  -8.478428999
  C    1.286787704  -8.127490488  -7.900832168
  N    2.226674897  -7.383557509  -7.174019467
  C    1.879401063  -6.083592144  -7.249561728
  N   -0.701588997  -7.586047546  -9.272905854
  C   -0.806148243  -8.939044932  -9.499258969
  N    0.065186832  -9.903034605  -8.930557396
  C    1.180380384  -9.552999190  -8.067416180
  N   -1.827356672  -9.368093258 -10.304512495
  H   -2.444073852  -8.686242192 -10.742847882
  H   -1.962754314 -10.337399294 -10.553964857
  O    1.850716620 -10.439529496  -7.591366533
  H    2.416819565  -5.205853501  -6.758312938
  H   -0.072526323 -10.908262518  -9.110655110
  P   -0.835378821  -1.333805007  -5.158687096
  O   -0.792446968   0.319306416  -5.108342486
  C    0.090170261   0.965950397  -4.165971565
  H    0.755295569   1.595278791  -4.805695659
  H    0.701198185   0.235315997  -3.595082460
  C   -0.731947686   1.856594017  -3.231568774
  H   -0.213043149   1.971873080  -2.244573630
  O   -0.736351973   3.196444191  -3.819842678
  C   -2.004824521   3.839301758  -3.627238156
  H   -1.778911096   4.810170877  -3.116638216
  C   -2.229613963   1.510656506  -3.026162852
  H   -2.668541574   0.953673683  -3.895458129
  C   -2.903487829   2.874939264  -2.844226398
  H   -2.944223410   3.135505505  -1.757991730
  H   -3.964394295   2.882117590  -3.159555832
  O   -2.376183294   0.784928516  -1.788345642
  O   -0.166045481  -2.129044002  -4.108922316
  O   -2.322525739  -1.477667706  -5.814832943
  N   -2.547491377   4.154256914  -4.983534492
  C   -2.807316336   3.314210255  -6.064942567
  C   -3.332082265   4.140813366  -7.113101409
  N   -3.378588606   5.475818770  -6.683462460
  C   -2.911447399   5.483056383  -5.438642411
  N   -2.671511364   1.942553284  -6.249755813
  C   -3.041994803   1.430893604  -7.504756517
  N   -3.532466847   2.156518518  -8.499217802
  C   -3.699784394   3.550359068  -8.364589462
  H   -2.920427259   0.329836675  -7.644171794
  N   -4.187585372   4.237035108  -9.411408312
  H   -4.430704463   3.773101883 -10.286090019
  H   -4.330454788   5.246213902  -9.357590119
  H   -2.800156680   6.354680403  -4.797145998
  P   -2.466168636  -0.838187933  -2.026440249
  O   -1.314806560  -1.425519971  -1.051065932
  C   -0.924781586  -2.798801409  -1.346512391
  H   -1.082825653  -3.062756759  -2.419717393
  H    0.181872466  -2.764818945  -1.158630554
  C   -1.576251562  -3.778679879  -0.368677280
  H   -1.276792990  -3.588977617   0.691911923
  O   -3.027216785  -3.617896207  -0.338033580
  C   -3.713998289  -4.818334277  -0.771798717
  H   -4.510989973  -4.973773572  -0.002817594
  C   -1.303827745  -5.245359980  -0.805082142
  H   -0.758355641  -5.308777490  -1.783405789
  C   -2.681191820  -5.932134909  -0.854126871
  H   -2.792635862  -6.561443787  -1.765899714
  H   -2.817944031  -6.656792608  -0.013045915
  O   -0.438049550  -5.756470924   0.227873237
  O   -2.355211156  -1.146433959  -3.503442205
  O   -3.885684599  -1.249217111  -1.390125370
  N   -4.351469139  -4.446935853  -2.072916574
  C   -3.556956774  -4.311594667  -3.254305944
  N   -4.103449498  -3.489066451  -4.264166760
  C   -5.319080180  -2.722454420  -4.150192830
  C   -6.174653504  -3.136786451  -3.026655781
  C   -5.675757910  -3.922912681  -2.042020713
  O   -2.469280870  -4.837530938  -3.408153166
  H   -3.507226790  -3.321534399  -5.091762309
  O   -5.458912554  -1.833418569  -4.962820033
  C   -7.575804089  -2.639361607  -3.030369689
  H   -7.757956240  -1.924522629  -2.211248740
  H   -7.827988490  -2.110323927  -3.966510843
  H   -8.307512115  -3.456821295  -2.927464054
  H   -6.274527099  -4.206834457  -1.164578230
  P   -0.033558201  -7.354252845   0.071678088
  O    0.723394460  -7.396737741  -1.386105664
  C    2.153002919  -7.259725559  -1.435486066
  H    2.656796429  -7.792188620  -0.599028538
  H    2.392876937  -7.784742891  -2.383454131
  C    2.566148147  -5.780569719  -1.453966171
  H    2.225779248  -5.213874140  -0.555527535
  O    4.026578155  -5.807199037  -1.344762822
  C    4.626227451  -5.378950228  -2.576501786
  H    5.615532854  -4.951864985  -2.297835902
  C    2.243593568  -5.015723304  -2.764920854
  H    1.778650905  -5.657634650  -3.541706686
  C    3.603745726  -4.430995334  -3.200912984
  H    3.695657553  -4.291265527  -4.295545883
  H    3.730391807  -3.410026444  -2.778091359
  O    1.318793070  -3.992609629  -2.412405253
  O   -1.148522075  -8.289884997   0.204870501
  O    1.216882981  -7.418442767   1.121172771
  N    4.835976120  -6.654160884  -3.373590738
  C    4.467866370  -6.773300757  -4.771415113
  N    4.118806369  -8.025773372  -5.254417176
  C    4.318140997  -9.156348520  -4.504038911
  C    4.934929107  -9.058654655  -3.201703163
  C    5.139382795  -7.820848607  -2.657642208
  O    4.420936130  -5.782162852  -5.488172997
  N    3.856545230 -10.329711263  -5.020801636
  H    4.012646862 -11.214423664  -4.558037880
  H    3.413963329 -10.366305941  -5.939769238
  H    3.913626976  -4.135431122  -6.202047477
  H    5.524224763  -7.681942993  -1.632107737
  H    5.225856440  -9.959868426  -2.661224346
  H    1.065244283  -3.448272597  -3.210800349
  H   -3.118834269  -1.566999046  -5.178635043
  H   -3.917610996  -2.195006188  -0.962955871
  H    1.035525360  -7.703179387   2.060174861
  H    3.050719869  -7.800288624  -6.640636401
  H   -2.179542280   1.336542575  -5.549974429

