%mem=64gb
%nproc=28       
%Chk=snap_77.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_77 

2     1 
  O    1.622552666  -4.177435453  -5.979282016
  C    1.828733058  -3.140256076  -6.938047091
  H    2.777951423  -3.443893385  -7.434546548
  H    1.967170195  -2.174085955  -6.416157153
  C    0.694185335  -3.049249826  -7.963190177
  H    0.714382674  -2.072278366  -8.500631352
  O    0.939017841  -4.044105994  -9.019138021
  C   -0.234494859  -4.798211802  -9.287650584
  H   -0.277986892  -4.944432421 -10.393371886
  C   -0.727239747  -3.365705374  -7.433921864
  H   -0.687845171  -4.024630662  -6.536267129
  C   -1.409937813  -4.090952438  -8.603510798
  H   -1.911091924  -3.365692803  -9.279019165
  H   -2.222625878  -4.769879642  -8.269616310
  O   -1.395916795  -2.123489957  -7.187304934
  N   -0.041313378  -6.159663720  -8.650959598
  C   -0.870671008  -7.274989809  -8.889053695
  C   -0.582997209  -8.224618135  -7.867137787
  N    0.415499915  -7.684171693  -7.045750187
  C    0.720644173  -6.446148503  -7.500474615
  N   -1.772247037  -7.460186932  -9.903780316
  C   -2.429933251  -8.662846706  -9.869567237
  N   -2.249383566  -9.630838858  -8.847756691
  C   -1.330091918  -9.448436067  -7.737136323
  N   -3.337113889  -8.918870941 -10.868200278
  H   -3.436090810  -8.253085145 -11.632478242
  H   -3.782490287  -9.816851077 -10.985337308
  O   -1.299515566 -10.287849092  -6.866346988
  H    1.475236081  -5.755115290  -7.040740655
  H   -2.811840460 -10.492762273  -8.836281997
  P   -1.712369099  -1.887146047  -5.565498109
  O   -2.430901230  -0.408600197  -5.719887529
  C   -1.577391739   0.743131126  -5.690057569
  H   -2.073910389   1.428175441  -6.410911151
  H   -0.548653778   0.537090196  -6.041932403
  C   -1.595590517   1.321897476  -4.264258652
  H   -0.696935358   1.037147609  -3.663007903
  O   -1.459770674   2.775338230  -4.392527293
  C   -2.610596481   3.457972606  -3.886770218
  H   -2.203379448   4.286640643  -3.251134545
  C   -2.926102595   1.056106117  -3.511848445
  H   -3.637387940   0.417630999  -4.095210866
  C   -3.502488896   2.444940882  -3.180288007
  H   -3.470457161   2.584257442  -2.068337031
  H   -4.574690505   2.506494773  -3.437431912
  O   -2.670132420   0.481158058  -2.225227137
  O   -0.451385775  -2.041750951  -4.800818231
  O   -2.973538991  -2.904603063  -5.462903775
  N   -3.278962295   4.082802194  -5.084101048
  C   -2.662447222   4.415451475  -6.284143123
  C   -3.646740938   5.070077510  -7.094095028
  N   -4.855924006   5.141851885  -6.389465141
  C   -4.635839845   4.564912237  -5.207567472
  N   -1.359709034   4.216624192  -6.741542664
  C   -1.075544728   4.703684145  -8.030280964
  N   -1.951698189   5.310638495  -8.814421350
  C   -3.281295193   5.541237893  -8.395818222
  H   -0.038653428   4.557736049  -8.407802778
  N   -4.109982825   6.183951331  -9.232724150
  H   -3.798656676   6.504829941 -10.150393435
  H   -5.077704291   6.377131419  -8.968536180
  H   -5.357960340   4.463963521  -4.401911781
  P   -2.299708917  -1.135217872  -2.146798519
  O   -0.763236419  -0.849385955  -1.676594594
  C    0.163682387  -1.969067276  -1.669641469
  H    0.192053745  -2.463941586  -2.673366434
  H    1.137685414  -1.479375932  -1.469223149
  C   -0.321200356  -2.858943741  -0.528520933
  H   -0.341924308  -2.321319082   0.452858680
  O   -1.737120115  -3.054515810  -0.879052724
  C   -2.044058440  -4.465358497  -1.050216098
  H   -3.042016898  -4.562574933  -0.554876892
  C    0.297727853  -4.259525971  -0.362207653
  H    1.068006032  -4.497304357  -1.149401817
  C   -0.902928106  -5.226352909  -0.374187867
  H   -0.650344569  -6.192803779  -0.861024281
  H   -1.191117154  -5.509518142   0.663258926
  O    0.914411886  -4.210567339   0.937572944
  O   -2.548218159  -1.834382159  -3.445921149
  O   -3.361145368  -1.488922315  -0.971413773
  N   -2.179953814  -4.778979307  -2.500207300
  C   -1.029273618  -4.972887572  -3.326022578
  N   -1.252581061  -5.558343519  -4.589414677
  C   -2.542548263  -5.862705115  -5.149125708
  C   -3.675538908  -5.514699515  -4.270969965
  C   -3.476873239  -5.015064501  -3.025862068
  O    0.102502826  -4.683442312  -2.974306894
  H   -0.407216268  -5.902159053  -5.075204090
  O   -2.553706262  -6.351162258  -6.258979427
  C   -5.031974321  -5.726647777  -4.843899926
  H   -5.026572101  -6.515710481  -5.617467976
  H   -5.779185594  -6.024647120  -4.094915368
  H   -5.399634132  -4.811207937  -5.339281201
  H   -4.315975011  -4.758514234  -2.363159215
  P    1.889874019  -5.494709436   1.318687781
  O    3.334480851  -4.743496332   1.148564346
  C    4.475358510  -5.559038293   0.781541946
  H    5.270254716  -5.150676998   1.447523150
  H    4.326385852  -6.636606275   1.001073701
  C    4.831394987  -5.345107810  -0.691292282
  H    5.942899904  -5.353253806  -0.817393389
  O    4.385868996  -6.522675704  -1.427534744
  C    3.689775765  -6.125629344  -2.625065012
  H    4.396105254  -6.303323291  -3.476912852
  C    4.193949013  -4.097327168  -1.365179167
  H    3.688853474  -3.422069495  -0.633117804
  C    3.269136678  -4.662403550  -2.450443130
  H    2.190956913  -4.575303738  -2.173775416
  H    3.336547374  -4.101400062  -3.406741745
  O    5.196073636  -3.246736959  -1.894667492
  O    1.647670249  -6.741303899   0.582203655
  O    1.682100043  -5.530322831   2.925803814
  N    2.530584363  -7.058618548  -2.796729129
  C    1.910452474  -7.063874943  -4.114093961
  N    0.945487388  -8.024092123  -4.378370793
  C    0.562377017  -8.925139234  -3.414952764
  C    1.115969878  -8.859247243  -2.094686111
  C    2.109661230  -7.948846361  -1.817251012
  O    2.245612823  -6.235412656  -4.944816665
  N   -0.387054566  -9.837128083  -3.794263106
  H   -0.650560590 -10.602776463  -3.192005009
  H   -0.674668467  -9.908456000  -4.771156259
  H    1.019161376  -3.888096306  -5.250037439
  H    2.607875020  -7.889905626  -0.826560539
  H    0.773656296  -9.536451893  -1.313276286
  H    5.714051832  -3.671079195  -2.613373853
  H   -3.339661247  -3.053081396  -4.517103771
  H   -3.012634542  -1.668459579  -0.045648771
  H    1.915443606  -4.710976864   3.451975099
  H    0.762003178  -8.131716405  -6.149604019
  H   -0.672847936   3.756922030  -6.122724721

