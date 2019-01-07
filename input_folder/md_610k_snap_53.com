%mem=64gb
%nproc=28       
%Chk=snap_53.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_53 

2     1 
  O    1.807102174  -5.844005395  -6.298123277
  C    2.786823817  -5.037200369  -6.972017392
  H    3.294342833  -5.684703137  -7.717178283
  H    3.521049340  -4.657385513  -6.235822388
  C    2.036207386  -3.895182020  -7.666010021
  H    2.690204085  -3.001970664  -7.825941471
  O    1.707667392  -4.316529355  -9.029536829
  C    0.307439978  -4.398303925  -9.228481280
  H    0.123533243  -3.997791556 -10.258374235
  C    0.698332275  -3.516348708  -6.987407164
  H    0.505310793  -4.132603532  -6.071883896
  C   -0.384821457  -3.685025829  -8.065477765
  H   -0.780289471  -2.697112189  -8.383852633
  H   -1.263744636  -4.239880629  -7.670559465
  O    0.856261459  -2.114516915  -6.656713114
  N   -0.057194355  -5.866461760  -9.256391766
  C   -0.590423006  -6.568785359 -10.360962807
  C   -0.731340978  -7.925043655  -9.955008631
  N   -0.278153976  -8.024960108  -8.623640970
  C    0.119721000  -6.790987983  -8.216232212
  N   -0.931881348  -6.067650675 -11.585704022
  C   -1.432156125  -6.998655202 -12.467417237
  N   -1.599875370  -8.363780355 -12.145462409
  C   -1.257978227  -8.932432827 -10.843418756
  N   -1.806980740  -6.551047243 -13.707984186
  H   -1.648186946  -5.576979327 -13.958286451
  H   -2.123073706  -7.167803254 -14.442249030
  O   -1.447206990 -10.108497434 -10.673588834
  H    0.535494987  -6.511023965  -7.199066221
  H   -1.981006649  -9.026593363 -12.837596829
  P    0.023367784  -1.557342397  -5.364013552
  O   -1.232793911  -0.721939749  -6.087180053
  C   -1.176295082   0.707860345  -5.981817840
  H   -1.979644824   1.029625167  -6.677191301
  H   -0.207887838   1.101170360  -6.348887297
  C   -1.456813409   1.153615803  -4.534674612
  H   -0.560573623   1.066162712  -3.855750675
  O   -1.712676163   2.593302389  -4.559936819
  C   -3.087779519   2.883410977  -4.295378894
  H   -3.082031704   3.665345250  -3.491125105
  C   -2.718631938   0.491274763  -3.936359804
  H   -3.021340102  -0.440299772  -4.479348996
  C   -3.807928886   1.581225892  -3.949064476
  H   -4.278780772   1.638590572  -2.936059275
  H   -4.625938722   1.314499806  -4.642062650
  O   -2.430800405   0.258476929  -2.545319802
  O    0.754092707  -0.716360496  -4.409255284
  O   -0.727676528  -2.888058359  -4.858882819
  N   -3.599017109   3.522102978  -5.560281209
  C   -2.817595604   4.167532831  -6.511867263
  C   -3.715088922   4.732695460  -7.475938682
  N   -5.035583956   4.436344969  -7.110859064
  C   -4.964005608   3.730028469  -5.981260370
  N   -1.435626530   4.314611844  -6.626798472
  C   -0.984795958   5.062430425  -7.727266705
  N   -1.774571855   5.606270201  -8.641034132
  C   -3.179386282   5.489927176  -8.566903963
  H    0.115023052   5.198964448  -7.832398712
  N   -3.916918193   6.095628764  -9.511026961
  H   -3.483882432   6.634351255 -10.261840979
  H   -4.936321756   6.053561544  -9.490461994
  H   -5.805550957   3.352780558  -5.405409981
  P   -2.412374038  -1.306418139  -2.032314845
  O   -0.977540963  -1.221766631  -1.274496626
  C   -0.373271186  -2.446337138  -0.797375086
  H    0.012090420  -3.026148962  -1.657864487
  H    0.487725286  -2.065163263  -0.194937552
  C   -1.393343502  -3.190618303   0.057443907
  H   -1.729848950  -2.600421262   0.945520796
  O   -2.582966407  -3.275803160  -0.804440647
  C   -2.918024780  -4.670737521  -1.095342051
  H   -4.036041185  -4.666451028  -1.068687836
  C   -1.033002795  -4.641259974   0.450382145
  H   -0.086261216  -5.007452337  -0.027980825
  C   -2.250999747  -5.478303521   0.014280404
  H   -1.961943439  -6.504797868  -0.284095278
  H   -2.934405812  -5.612605992   0.886800149
  O   -0.969309745  -4.737997984   1.875322186
  O   -2.585471849  -2.258290439  -3.184471495
  O   -3.666735427  -1.172757740  -1.012756114
  N   -2.471660652  -5.010305152  -2.475096321
  C   -1.176200531  -5.554513154  -2.762288709
  N   -0.935971894  -5.913881157  -4.102663446
  C   -1.819034844  -5.652603879  -5.196516070
  C   -3.024428845  -4.898979546  -4.843511188
  C   -3.310082562  -4.601938690  -3.549798484
  O   -0.329341637  -5.749006854  -1.903732052
  H   -0.009773178  -6.358624737  -4.302816234
  O   -1.486608535  -6.086503629  -6.288901501
  C   -3.865454513  -4.414875374  -5.973783038
  H   -3.532636515  -3.412364153  -6.292139716
  H   -3.797188806  -5.082903992  -6.847332768
  H   -4.930646300  -4.341321481  -5.715285357
  H   -4.204483476  -4.026122677  -3.271205804
  P    0.506134142  -4.510500061   2.597836575
  O    1.301290148  -5.937223655   2.431803365
  C    1.681600267  -6.470884047   1.156755913
  H    1.676656249  -7.568333049   1.355142180
  H    0.948647283  -6.246750725   0.354800246
  C    3.106951614  -5.994069548   0.840897246
  H    3.667260231  -5.796819929   1.787058519
  O    3.824628861  -7.109648802   0.241318021
  C    4.089237972  -6.856801995  -1.146546883
  H    5.057888641  -7.369437535  -1.356759795
  C    3.192431858  -4.821533144  -0.175168073
  H    2.197224454  -4.482429536  -0.552435650
  C    4.085009778  -5.337900154  -1.311010263
  H    3.734470745  -4.992793924  -2.312290154
  H    5.115696812  -4.934526611  -1.244424219
  O    3.673669271  -3.644295277   0.463970469
  O    0.463796604  -4.026913236   3.963052786
  O    1.151161070  -3.564005896   1.404139061
  N    2.987727193  -7.580821468  -1.896833757
  C    2.467362241  -7.141404979  -3.180288831
  N    1.396123986  -7.827420345  -3.736186072
  C    0.852867288  -8.903299180  -3.095029822
  C    1.411018455  -9.403853173  -1.867620425
  C    2.458720792  -8.727495145  -1.303249087
  O    2.975913564  -6.188019605  -3.759625194
  N   -0.266475831  -9.457837674  -3.664203787
  H   -0.668419964 -10.313962280  -3.317360393
  H   -0.635074011  -9.089823568  -4.528123359
  H    2.153135321  -6.213902442  -5.428321669
  H    2.921778454  -9.057452055  -0.355030145
  H    1.013696105 -10.303737558  -1.401564242
  H    4.606530141  -3.727505785   0.770208155
  H   -1.442162754  -2.807028230  -4.062968322
  H   -3.849235611  -1.924371647  -0.364900327
  H    2.117141638  -3.268244007   1.460144400
  H   -0.275642683  -8.887382043  -8.092696378
  H   -0.820851441   3.908526183  -5.899252830

