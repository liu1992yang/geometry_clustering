%mem=64gb
%nproc=28       
%Chk=snap_164.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_164 

2     1 
  O    3.301755026  -4.266941231  -4.468158573
  C    3.607054491  -3.010416008  -5.087315480
  H    4.604047427  -3.195571980  -5.546100965
  H    3.705722305  -2.248843673  -4.287074838
  C    2.590054285  -2.568648476  -6.142276475
  H    2.771754496  -1.511172780  -6.468637464
  O    2.806067799  -3.330425579  -7.367395478
  C    1.671016736  -4.114637194  -7.703659218
  H    1.495339853  -3.943634158  -8.798092630
  C    1.113980007  -2.777277020  -5.732991885
  H    1.013327744  -3.150910455  -4.683127967
  C    0.521911318  -3.750260988  -6.761536533
  H   -0.325440332  -3.288303953  -7.314945380
  H    0.080607329  -4.637101165  -6.248903861
  O    0.523663016  -1.461958749  -5.871483895
  N    2.081859091  -5.558602495  -7.544415806
  C    1.862007886  -6.571350649  -8.495894902
  C    2.397989188  -7.774209804  -7.947653582
  N    2.939826441  -7.466685072  -6.682514353
  C    2.747453605  -6.147049573  -6.450890981
  N    1.250326388  -6.450107360  -9.714915797
  C    1.187436744  -7.610358615 -10.443853102
  N    1.697298021  -8.847484874  -9.976564630
  C    2.345406780  -9.007562106  -8.685206946
  N    0.587641780  -7.558605144 -11.674768071
  H    0.249965480  -6.666111722 -12.030611279
  H    0.516258692  -8.356534877 -12.288908002
  O    2.740532529 -10.118943720  -8.407345868
  H    3.076336800  -5.540728812  -5.540145505
  H    1.635204118  -9.697007270 -10.557117856
  P   -0.792748181  -1.187084608  -4.924022198
  O   -1.267438532   0.270549183  -5.551185697
  C   -0.541999727   1.445829219  -5.108097122
  H   -0.492381121   2.069107356  -6.033454577
  H    0.488938404   1.195989090  -4.788646005
  C   -1.314520975   2.173798050  -4.007754391
  H   -0.620055238   2.599726787  -3.241324079
  O   -1.959004828   3.326587915  -4.642731140
  C   -3.270257411   3.539428503  -4.099811713
  H   -3.294891762   4.612450770  -3.782473399
  C   -2.476471789   1.404177098  -3.330702981
  H   -2.948017792   0.663614916  -4.033982156
  C   -3.494018355   2.504140894  -2.994648394
  H   -3.301133644   2.922152617  -1.981656329
  H   -4.539709136   2.123816652  -2.940816156
  O   -1.954188375   0.790526837  -2.156973130
  O   -0.400750958  -1.201054206  -3.477278010
  O   -1.851381754  -2.150168093  -5.644862345
  N   -4.238902850   3.381590264  -5.233421746
  C   -4.425542235   2.328459901  -6.123373693
  C   -5.516028043   2.698921491  -6.981887134
  N   -5.990903600   3.968024774  -6.619011303
  C   -5.239974385   4.365323038  -5.594492676
  N   -3.783658348   1.104343490  -6.298005062
  C   -4.227842223   0.297121117  -7.353985443
  N   -5.247938073   0.586324812  -8.148421923
  C   -5.946917288   1.804706681  -8.009156259
  H   -3.679182668  -0.661407097  -7.509963670
  N   -6.968023222   2.052581093  -8.848846234
  H   -7.238566282   1.387836444  -9.570415898
  H   -7.500152413   2.920133218  -8.780421948
  H   -5.329336933   5.308328023  -5.059386352
  P   -2.233455176  -0.853802564  -2.105362686
  O   -1.088645913  -1.345782121  -1.068523656
  C   -0.526681777  -2.662013645  -1.262994270
  H    0.171532853  -2.645559299  -2.126747723
  H    0.071770560  -2.779235774  -0.330718595
  C   -1.643041736  -3.706589172  -1.358087370
  H   -1.648802553  -4.380209682  -0.466464430
  O   -2.892809612  -2.953148217  -1.335595560
  C   -3.960903485  -3.806797450  -1.921996357
  H   -4.464036267  -4.269299603  -1.034989919
  C   -1.747474969  -4.538452007  -2.662697462
  H   -1.321537556  -4.043490075  -3.562526539
  C   -3.247709228  -4.827373831  -2.810509758
  H   -3.573890415  -4.804251373  -3.877152097
  H   -3.464090130  -5.872321325  -2.487514332
  O   -1.128687840  -5.824571461  -2.411031798
  O   -2.661521927  -1.320827858  -3.492781605
  O   -3.523167541  -0.739906470  -1.083729183
  N   -4.911560860  -2.868840945  -2.560653867
  C   -4.730284868  -2.424760155  -3.904329097
  N   -5.331204949  -1.200730665  -4.266626888
  C   -5.944788216  -0.285972146  -3.337751952
  C   -6.229721491  -0.876113855  -2.023218092
  C   -5.689185946  -2.066945782  -1.667608148
  O   -4.178025821  -3.108585590  -4.766127835
  H   -5.184130149  -0.880627120  -5.226251552
  O   -6.121462585   0.845792767  -3.741327398
  C   -7.080571781  -0.091025822  -1.088366399
  H   -6.527554397   0.184913989  -0.175220808
  H   -7.437155073   0.851876234  -1.537307913
  H   -7.979003278  -0.649881176  -0.779520218
  H   -5.835474053  -2.493176247  -0.666429973
  P    0.420497002  -5.957595546  -2.912628128
  O    1.231840902  -5.285170155  -1.646819533
  C    2.638004426  -5.618406882  -1.545319968
  H    3.090809406  -5.944993991  -2.511351851
  H    3.097107152  -4.646269051  -1.257166946
  C    2.797907663  -6.656742938  -0.434614676
  H    2.133226423  -6.443068362   0.442649926
  O    2.319805189  -7.960978042  -0.890405557
  C    3.379725014  -8.954658950  -0.862491130
  H    3.032585740  -9.688145178  -0.090042670
  C    4.275795817  -6.841488056  -0.006373513
  H    4.960102556  -6.050943104  -0.382001274
  C    4.682699850  -8.242937789  -0.494282927
  H    5.380360870  -8.174916575  -1.355926389
  H    5.233215780  -8.796011647   0.293679007
  O    4.210733865  -6.776342263   1.418678204
  O    0.741804398  -5.191195220  -4.152612819
  O    0.645358392  -7.542961004  -2.932979647
  N    3.382138437  -9.599712915  -2.206169732
  C    3.602273627  -8.834407376  -3.439782062
  N    3.521891546  -9.500863969  -4.647240618
  C    3.333771845 -10.859128726  -4.696437480
  C    3.161680400 -11.628996083  -3.482599599
  C    3.176350897 -10.984465125  -2.281601233
  O    3.809648995  -7.634973678  -3.357205310
  N    3.314153766 -11.437080598  -5.926248920
  H    3.167522444 -12.427511258  -6.054610330
  H    3.395114383 -10.870310804  -6.775989367
  H    2.351066550  -4.296327475  -4.129099690
  H    3.032521025 -11.529140955  -1.334860041
  H    3.016668488 -12.709066185  -3.534779345
  H    5.103785653  -6.716050115   1.825594383
  H   -2.724503920  -2.407835679  -5.118300683
  H   -3.360581480  -1.003939090  -0.134512128
  H    1.414192212  -7.928658570  -2.325039210
  H    3.353062817  -8.174428103  -6.026576791
  H   -2.964022260   0.819058168  -5.704873989
