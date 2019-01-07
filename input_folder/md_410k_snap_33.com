%mem=64gb
%nproc=28       
%Chk=snap_33.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_33 

2     1 
  O    2.072895983  -3.725415160  -6.352082046
  C    2.421427772  -3.371945435  -7.684855933
  H    2.714288912  -4.336119333  -8.149260522
  H    3.308827446  -2.708168597  -7.672199908
  C    1.248932701  -2.722577863  -8.436957711
  H    1.424309358  -1.639714699  -8.641431778
  O    1.216373012  -3.339556265  -9.767933786
  C   -0.079499092  -3.818020893 -10.088189304
  H   -0.268016375  -3.518954423 -11.150928837
  C   -0.158210362  -2.944689759  -7.826759835
  H   -0.166232421  -3.750382617  -7.056260969
  C   -1.051950438  -3.311447442  -9.025678859
  H   -1.599368851  -2.404736112  -9.370519554
  H   -1.846278172  -4.029867064  -8.747891197
  O   -0.605335847  -1.683473539  -7.317085584
  N   -0.004221887  -5.331499355 -10.066922076
  C   -0.222183653  -6.183231529 -11.173034052
  C   -0.080083365  -7.518374296 -10.696222427
  N    0.228277564  -7.456791841  -9.326004170
  C    0.274332855  -6.148900630  -8.959998177
  N   -0.529191180  -5.824301393 -12.453530606
  C   -0.691506527  -6.880120647 -13.324204810
  N   -0.568966239  -8.231999519 -12.934719980
  C   -0.256917122  -8.655793143 -11.571065264
  N   -1.005134214  -6.580402027 -14.622939581
  H   -1.069621328  -5.606405127 -14.913196386
  H   -1.097024183  -7.281444654 -15.344067984
  O   -0.191009676  -9.834458907 -11.345757414
  H    0.463606654  -5.768367986  -7.930893637
  H   -0.705895125  -8.992089600 -13.618750349
  P   -0.803550660  -1.634394119  -5.660309433
  O   -1.442255740  -0.107435855  -5.665777298
  C   -0.552253001   0.910053229  -5.140457072
  H   -0.400313284   1.622692900  -5.981359294
  H    0.438539958   0.503945512  -4.829904706
  C   -1.265591728   1.590751047  -3.969698191
  H   -0.584530805   1.743323326  -3.094436446
  O   -1.585111976   2.960012761  -4.389593486
  C   -2.964053384   3.244251457  -4.215523105
  H   -3.003390002   4.264248552  -3.769978523
  C   -2.609412629   0.937633555  -3.544752696
  H   -2.989378447   0.178885682  -4.273779835
  C   -3.583824330   2.117888676  -3.388463799
  H   -3.650346959   2.389182528  -2.306836828
  H   -4.616993982   1.853574242  -3.677328919
  O   -2.473910147   0.361947743  -2.244221609
  O    0.493920307  -1.819870051  -4.966479211
  O   -2.011740322  -2.720254934  -5.614147806
  N   -3.560880906   3.291900361  -5.599529366
  C   -4.239548572   4.345669284  -6.202672301
  C   -4.534647521   3.931849969  -7.544178660
  N   -4.030323096   2.641021596  -7.760099480
  C   -3.452534510   2.271319963  -6.617522329
  N   -4.656411669   5.593547633  -5.733918044
  C   -5.362632146   6.401909594  -6.648883629
  N   -5.643644053   6.058994566  -7.893231605
  C   -5.243781216   4.811805026  -8.420320307
  H   -5.705859481   7.396828288  -6.282345282
  N   -5.551636618   4.533092994  -9.697353840
  H   -6.067961342   5.194434103 -10.277884253
  H   -5.275799815   3.647792641 -10.122780251
  H   -2.937202362   1.323277148  -6.426151483
  P   -1.771207492  -1.147575069  -2.196153982
  O   -0.372956728  -0.591573513  -1.569268271
  C    0.728925347  -1.531369531  -1.441152908
  H    0.972108727  -1.995224976  -2.428555227
  H    1.567177561  -0.881928746  -1.118227556
  C    0.270168150  -2.521610663  -0.375162170
  H    0.002387180  -2.022279937   0.589760570
  O   -1.011145227  -2.990420217  -0.924256827
  C   -0.987727842  -4.433078980  -1.168305656
  H   -1.969191656  -4.773131166  -0.749886062
  C    1.142079729  -3.765026832  -0.114178896
  H    2.095079662  -3.765443898  -0.698663426
  C    0.235515126  -4.961803919  -0.428102708
  H    0.808560485  -5.718194025  -1.023857150
  H   -0.104856086  -5.512745707   0.488277884
  O    1.505302798  -3.631904745   1.276195180
  O   -1.693543606  -1.771796782  -3.555847739
  O   -2.891587788  -1.753183787  -1.195303022
  N   -0.963163260  -4.666433629  -2.635555698
  C    0.276385577  -4.597773795  -3.358531609
  N    0.233858656  -5.030575244  -4.701482576
  C   -0.910592900  -5.571221013  -5.350326003
  C   -2.123588994  -5.641790546  -4.532548170
  C   -2.108725798  -5.245425924  -3.233957138
  O    1.302298013  -4.182270677  -2.853061934
  H    1.128156852  -4.951674620  -5.240004345
  O   -0.778438949  -5.888608428  -6.528347112
  C   -3.341307656  -6.177687153  -5.200412882
  H   -4.121207817  -6.488963047  -4.492624680
  H   -3.778920022  -5.429943903  -5.880526659
  H   -3.102327204  -7.066220296  -5.809618187
  H   -2.987984563  -5.349811163  -2.583091530
  P    1.882653357  -4.984298933   2.132727708
  O    3.161032154  -5.559087912   1.297304780
  C    3.686438405  -6.835180061   1.771267020
  H    4.654222339  -6.567228383   2.251384354
  H    3.022461094  -7.334914992   2.506157091
  C    3.902094758  -7.734742703   0.551829360
  H    4.794723793  -8.388356675   0.715979789
  O    2.808593980  -8.691855216   0.466637077
  C    2.035801279  -8.514751272  -0.733865604
  H    2.054300132  -9.509924801  -1.242828014
  C    3.967705885  -6.992212387  -0.808190818
  H    4.116949955  -5.890499117  -0.696490829
  C    2.666736028  -7.369132681  -1.532290943
  H    1.990218694  -6.487221289  -1.618357718
  H    2.847914245  -7.653682686  -2.588359608
  O    5.124842421  -7.376184855  -1.529556692
  O    0.776260895  -5.929779468   2.368646729
  O    2.572620723  -4.300836752   3.437718024
  N    0.616233294  -8.239528101  -0.312032781
  C   -0.377601879  -7.868288088  -1.339940764
  N   -1.684819583  -7.673976194  -0.934362345
  C   -2.046184062  -7.900858176   0.364392683
  C   -1.091228998  -8.267601030   1.366973652
  C    0.225779371  -8.416524054   1.009084494
  O    0.038042968  -7.692181160  -2.472347870
  N   -3.372435807  -7.675274358   0.680924381
  H   -3.736253374  -7.986161898   1.570432533
  H   -4.050126659  -7.585274234  -0.065315156
  H    1.881359172  -2.910832839  -5.782378467
  H    1.007303105  -8.677207873   1.742140567
  H   -1.388041751  -8.385616473   2.407256368
  H    5.130986242  -8.331136309  -1.757076526
  H   -2.287107981  -3.038397505  -4.674308501
  H   -2.605832380  -2.120117690  -0.298996558
  H    1.999382704  -4.099333441   4.228827190
  H    0.362841506  -8.271713403  -8.731818867
  H   -4.441193062   5.912637382  -4.787172425

