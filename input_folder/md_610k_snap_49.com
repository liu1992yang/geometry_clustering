%mem=64gb
%nproc=28       
%Chk=snap_49.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_49 

2     1 
  O    2.113408615  -4.439337853  -5.840753381
  C    2.601609276  -3.360172019  -6.652117572
  H    3.633196085  -3.696173499  -6.901114891
  H    2.663599671  -2.438038378  -6.032833261
  C    1.785214444  -3.124441072  -7.924020205
  H    2.095527274  -2.168003530  -8.415642801
  O    2.139842209  -4.152203687  -8.911007416
  C    0.976209497  -4.755054379  -9.464582300
  H    1.174669398  -4.893038952 -10.554859652
  C    0.245229138  -3.220054559  -7.769385859
  H   -0.063178641  -3.825975326  -6.877612514
  C   -0.228895081  -3.896242321  -9.061684421
  H   -0.478943788  -3.139531585  -9.834738706
  H   -1.161073794  -4.479303206  -8.903660050
  O   -0.350082250  -1.906304358  -7.714447261
  N    0.871157072  -6.127369085  -8.837955204
  C    0.140242670  -7.220645344  -9.340934041
  C    0.149475101  -8.222042324  -8.326164345
  N    0.868716490  -7.716534980  -7.229338624
  C    1.278296281  -6.464731948  -7.531186978
  N   -0.462963509  -7.350421140 -10.565872054
  C   -1.097641644  -8.547225906 -10.769552895
  N   -1.149141337  -9.584831286  -9.802009400
  C   -0.532392314  -9.476405570  -8.492095636
  N   -1.755343530  -8.727173123 -11.962870683
  H   -1.679121077  -8.010202063 -12.682849009
  H   -2.123777567  -9.621943823 -12.252631919
  O   -0.658583048 -10.403902618  -7.721747521
  H    1.865235565  -5.767335948  -6.843441365
  H   -1.663910951 -10.456170556  -9.993044314
  P   -0.238280909  -1.232552524  -6.210391925
  O   -1.560015158  -0.233979345  -6.202362677
  C   -1.251900719   1.163445058  -6.029578844
  H   -2.141307707   1.655587012  -6.474683347
  H   -0.344429972   1.470462633  -6.590003052
  C   -1.117212694   1.454639714  -4.524278233
  H   -0.116774933   1.179135613  -4.098091248
  O   -1.163244975   2.903693576  -4.341428981
  C   -2.398160565   3.310010364  -3.746923704
  H   -2.122506200   4.109015194  -3.010829908
  C   -2.286801918   0.862456111  -3.702240769
  H   -2.910991695   0.153087169  -4.304686379
  C   -3.081077320   2.072193323  -3.171577713
  H   -3.026829892   2.076893130  -2.053061019
  H   -4.157431488   1.987102964  -3.400571649
  O   -1.662447054   0.221946652  -2.576163407
  O    1.022988170  -0.535640853  -5.916305750
  O   -0.609657380  -2.497323204  -5.284905814
  N   -3.172667845   3.954373337  -4.869267396
  C   -2.602714837   4.520229772  -6.006162493
  C   -3.674431038   5.061830501  -6.786307810
  N   -4.890120219   4.833461215  -6.126891641
  C   -4.591538396   4.185118901  -4.999546175
  N   -1.275877163   4.609803399  -6.427502266
  C   -1.056598781   5.276713354  -7.645087382
  N   -2.014056614   5.792170675  -8.400203258
  C   -3.372321060   5.724842834  -8.020120961
  H   -0.001686437   5.365209556  -7.991625522
  N   -4.285948825   6.277402091  -8.832645363
  H   -4.015706394   6.748782452  -9.696981411
  H   -5.279286055   6.258846948  -8.598268045
  H   -5.299269567   3.867582510  -4.237796481
  P   -2.150000310  -1.324765422  -2.248461123
  O   -0.783426540  -1.827118099  -1.537118837
  C   -0.845802217  -2.098385481  -0.118254917
  H    0.200315994  -2.438079957   0.092899146
  H   -1.023123222  -1.158474749   0.441049698
  C   -1.896307693  -3.174458076   0.166574851
  H   -2.539755552  -2.923258255   1.043390886
  O   -2.794242432  -3.170263193  -1.002041353
  C   -3.079495143  -4.549791337  -1.453027510
  H   -4.196647109  -4.553032726  -1.511173554
  C   -1.374549118  -4.628784074   0.268238082
  H   -0.388098630  -4.752260137  -0.253896349
  C   -2.491885470  -5.463393691  -0.385452922
  H   -2.118536168  -6.430850853  -0.774590390
  H   -3.246502872  -5.743138344   0.386401027
  O   -1.336029953  -5.014712575   1.639827119
  O   -2.494182606  -2.026575526  -3.530256243
  O   -3.391092196  -0.922548819  -1.273392717
  N   -2.520122849  -4.741540259  -2.811338526
  C   -1.203072165  -5.264989329  -3.041814464
  N   -0.864048111  -5.514530355  -4.384600909
  C   -1.691480831  -5.211731956  -5.514177790
  C   -2.944204513  -4.521865181  -5.197595715
  C   -3.309190847  -4.295915452  -3.911595709
  O   -0.426308246  -5.537413582  -2.140264477
  H    0.057216542  -5.986857578  -4.559590809
  O   -1.258321171  -5.531400411  -6.607660434
  C   -3.737000800  -4.039323162  -6.362031387
  H   -3.754561641  -4.786450924  -7.172765683
  H   -4.780860127  -3.811311248  -6.110143573
  H   -3.290197948  -3.120606789  -6.779618211
  H   -4.233761844  -3.761019976  -3.656101149
  P    0.136233031  -5.057726211   2.403728770
  O    0.847677929  -6.475504913   1.988845002
  C    1.120469652  -6.901312807   0.647388642
  H    1.183129713  -8.008631483   0.770437002
  H    0.301951658  -6.673012540  -0.057218629
  C    2.483101259  -6.334770239   0.226159888
  H    3.125132810  -6.163703559   1.126236390
  O    3.209521654  -7.361839918  -0.500373749
  C    3.361354887  -7.018146819  -1.884982775
  H    4.402118914  -7.336983183  -2.144425472
  C    2.407552748  -5.099931211  -0.708679696
  H    1.364575427  -4.723594807  -0.880883662
  C    3.091112010  -5.521174394  -2.013243401
  H    2.467679989  -5.258296094  -2.905093540
  H    4.029448588  -4.958012812  -2.187949561
  O    3.023143955  -3.985532356  -0.061233865
  O    0.083712443  -4.861583987   3.838352470
  O    0.849364037  -3.922447801   1.442923033
  N    2.407678474  -7.935980279  -2.625385272
  C    1.785033103  -7.598312928  -3.886035996
  N    1.077376989  -8.581510289  -4.562412283
  C    0.918907659  -9.840569834  -4.039961528
  C    1.484217872 -10.164104976  -2.754418825
  C    2.223405876  -9.218696065  -2.098064596
  O    1.869811408  -6.466843571  -4.357306510
  N    0.218230886 -10.734208600  -4.793102701
  H    0.073151661 -11.687071697  -4.493012181
  H   -0.133907077 -10.486736124  -5.721758730
  H    1.226117298  -4.238637421  -5.458596639
  H    2.705698634  -9.425709446  -1.123705923
  H    1.345062033 -11.155775986  -2.324284792
  H    3.998467666  -4.090597631   0.030057634
  H   -1.337182395  -2.407436328  -4.500495790
  H   -3.963020169  -1.663662178  -0.890843884
  H    1.847646650  -3.822242423   1.374119221
  H    0.971428714  -8.206305731  -6.299166748
  H   -0.527033816   4.213490465  -5.833476456

