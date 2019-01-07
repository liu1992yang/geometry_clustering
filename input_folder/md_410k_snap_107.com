%mem=64gb
%nproc=28       
%Chk=snap_107.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_107 

2     1 
  O    3.030687174  -5.064183750  -7.115295096
  C    2.864503695  -3.804810427  -7.771260226
  H    3.470959416  -3.944215527  -8.698505749
  H    3.318324820  -3.010253173  -7.148049750
  C    1.422941018  -3.450721684  -8.140829667
  H    1.388474377  -2.418359702  -8.575279570
  O    1.026107854  -4.323795093  -9.253791389
  C   -0.259627918  -4.887558982  -9.032788672
  H   -0.785392418  -4.888004101 -10.017518402
  C    0.315578256  -3.602335988  -7.065193643
  H    0.600259810  -4.287772369  -6.233409510
  C   -0.894013363  -4.110554104  -7.874690687
  H   -1.503344697  -3.261086454  -8.249873107
  H   -1.606126142  -4.712363477  -7.269028133
  O    0.120637678  -2.271432981  -6.556646586
  N   -0.055088014  -6.327323010  -8.616386729
  C   -1.056154838  -7.325294920  -8.603251576
  C   -0.594281406  -8.350813924  -7.728300069
  N    0.670842644  -7.973241704  -7.244365077
  C    0.981221341  -6.765771459  -7.770971898
  N   -2.226104526  -7.360572936  -9.312643480
  C   -3.002659373  -8.463922583  -9.069366739
  N   -2.638972110  -9.503787657  -8.179598265
  C   -1.407279665  -9.492647481  -7.402145040
  N   -4.222938196  -8.540921239  -9.705488671
  H   -4.456301005  -7.835148778 -10.401150131
  H   -4.750406402  -9.401298542  -9.748708337
  O   -1.215063035 -10.387727938  -6.614205807
  H    1.951527467  -6.162048187  -7.615572383
  H   -3.281631105 -10.285672948  -7.995920663
  P   -0.812107150  -2.088610572  -5.195905165
  O   -1.734791703  -0.809789611  -5.704748787
  C   -1.281107028   0.502647130  -5.350559024
  H   -1.737348836   1.130708203  -6.146132044
  H   -0.180014857   0.599170556  -5.402790153
  C   -1.828020085   0.875356979  -3.959714119
  H   -1.143935909   0.578586278  -3.126644167
  O   -1.817181361   2.335805858  -3.869787255
  C   -3.140549118   2.866891438  -3.739219161
  H   -3.102687860   3.535709068  -2.839136841
  C   -3.294350169   0.416365027  -3.756669255
  H   -3.641022718  -0.279465230  -4.562004894
  C   -4.129747420   1.706351115  -3.670616172
  H   -4.689658261   1.704629317  -2.700427230
  H   -4.911109031   1.725599728  -4.453283195
  O   -3.493981940  -0.190368965  -2.470439919
  O    0.052499369  -1.840741092  -4.019822875
  O   -1.792724236  -3.359896646  -5.426874448
  N   -3.356663922   3.733765100  -4.949418376
  C   -2.377910340   4.246783726  -5.791047266
  C   -3.044332125   5.087400340  -6.741836899
  N   -4.420259795   5.093143423  -6.475477228
  C   -4.601403879   4.301232674  -5.418796197
  N   -0.995518303   4.065259632  -5.822425742
  C   -0.303908706   4.761382225  -6.830446900
  N   -0.876301865   5.544665461  -7.729825026
  C   -2.271261659   5.768248222  -7.736345605
  H    0.800479745   4.630080541  -6.871268676
  N   -2.777089198   6.596348637  -8.662351997
  H   -2.178570129   7.062597419  -9.345402993
  H   -3.778608677   6.790873543  -8.705135061
  H   -5.547751536   4.091001617  -4.926177388
  P   -2.836655429  -1.687504340  -2.207590338
  O   -1.721075757  -1.164425531  -1.140412168
  C   -0.568424923  -2.034674721  -0.975379760
  H   -0.130914138  -2.290438286  -1.971110845
  H    0.144493796  -1.396152121  -0.410213755
  C   -1.071069693  -3.230919051  -0.175052767
  H   -1.350771139  -2.961878825   0.870925715
  O   -2.337921019  -3.563305087  -0.846826950
  C   -2.452437857  -5.010780043  -1.057459055
  H   -3.457792413  -5.248744320  -0.629007998
  C   -0.225831195  -4.520473400  -0.217525292
  H    0.491526690  -4.524611307  -1.082726488
  C   -1.275680025  -5.642784903  -0.316832333
  H   -0.860365959  -6.551089890  -0.802749478
  H   -1.568283998  -5.980122431   0.702799682
  O    0.441340441  -4.623576285   1.048808594
  O   -2.369250314  -2.299527399  -3.490539865
  O   -4.148716332  -2.366508927  -1.533318383
  N   -2.470025844  -5.271764708  -2.519832186
  C   -1.272933672  -5.586170836  -3.245571742
  N   -1.441933303  -6.083905925  -4.550988918
  C   -2.693222262  -6.140579255  -5.249709538
  C   -3.839518563  -5.602664443  -4.492371058
  C   -3.713011391  -5.214150918  -3.200209664
  O   -0.163617263  -5.457311872  -2.756417118
  H   -0.568113121  -6.386194282  -5.034416714
  O   -2.680229654  -6.584167471  -6.378183184
  C   -5.128999827  -5.518967273  -5.229705915
  H   -5.831730641  -4.798507843  -4.790179771
  H   -4.975292784  -5.221604542  -6.281317548
  H   -5.635930960  -6.498679515  -5.254691769
  H   -4.560058979  -4.820448554  -2.618379645
  P    2.015101279  -5.142054478   0.949466054
  O    2.614316690  -3.703743243   0.420875428
  C    3.937295357  -3.681583184  -0.161335566
  H    4.394143931  -2.794326763   0.335588365
  H    4.531322304  -4.587707004   0.077964437
  C    3.865252863  -3.473624744  -1.675909133
  H    4.672845090  -2.773571569  -2.012184605
  O    4.202515212  -4.757979626  -2.296394204
  C    3.416999712  -4.977676717  -3.475673053
  H    4.153200667  -5.025093828  -4.317229091
  C    2.507268005  -3.051027226  -2.274479520
  H    1.658843000  -3.224427956  -1.576836892
  C    2.369990542  -3.867013179  -3.568483009
  H    1.334000934  -4.262612999  -3.680323357
  H    2.530765742  -3.222943196  -4.453888204
  O    2.595031824  -1.652155463  -2.541514772
  O    2.340003244  -6.324523466   0.149874844
  O    2.363223506  -5.201339381   2.529517441
  N    2.788438104  -6.346249855  -3.363730559
  C    1.867690942  -6.763654974  -4.390826096
  N    1.418012891  -8.059304377  -4.431376404
  C    1.739479948  -8.925929027  -3.405578285
  C    2.536863423  -8.482559528  -2.299648768
  C    3.045904783  -7.206039831  -2.294494404
  O    1.500844382  -5.941034363  -5.251583263
  N    1.250227189 -10.194460198  -3.527660005
  H    1.455820857 -10.912167419  -2.848321343
  H    0.680729776 -10.458503116  -4.326850206
  H    2.632207107  -5.069374324  -6.191683012
  H    3.674315637  -6.811846126  -1.470094390
  H    2.727564969  -9.138758402  -1.448771566
  H    1.820384152  -1.376055183  -3.108125978
  H   -2.469543227  -3.540069546  -4.663425944
  H   -4.146355638  -2.483477772  -0.531850934
  H    1.986176205  -4.493243354   3.128231706
  H    1.150797607  -8.444856608  -6.465685469
  H   -0.547257114   3.488692309  -5.092463881

