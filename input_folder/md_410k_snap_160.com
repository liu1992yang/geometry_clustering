%mem=64gb
%nproc=28       
%Chk=snap_160.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_160 

2     1 
  O    1.796288102  -2.621985717  -4.709728515
  C    1.384823887  -3.691726504  -5.551787157
  H    0.810798409  -4.430427277  -4.959819228
  H    2.353063711  -4.165677839  -5.843316666
  C    0.641976993  -3.161314137  -6.783487552
  H    0.894225043  -2.087493865  -6.963888199
  O    1.183683232  -3.846144032  -7.965525807
  C    0.157108417  -4.465857359  -8.726859685
  H    0.422553606  -4.325245224  -9.801641800
  C   -0.889194786  -3.406950301  -6.818130471
  H   -1.247163384  -4.120093204  -6.043571218
  C   -1.181147933  -3.907513412  -8.241567333
  H   -1.519764109  -3.058990274  -8.878965459
  H   -2.022724858  -4.630027198  -8.282359621
  O   -1.575800730  -2.144024482  -6.702560478
  N    0.263033197  -5.945118060  -8.412227912
  C   -0.687488747  -6.952921947  -8.684915745
  C   -0.287027093  -8.095028557  -7.930701644
  N    0.904959158  -7.785093936  -7.264351023
  C    1.218320020  -6.494955343  -7.533539809
  N   -1.776413348  -6.898533880  -9.509067329
  C   -2.537904020  -8.044961553  -9.523709480
  N   -2.239902540  -9.201991222  -8.762945149
  C   -1.120760485  -9.268753296  -7.839430607
  N   -3.652815615  -8.046888762 -10.323192932
  H   -3.840788760  -7.242779537 -10.917999708
  H   -4.223120880  -8.867162767 -10.467960454
  O   -1.012522997 -10.237237847  -7.123565557
  H    2.093406391  -5.943107028  -7.142639342
  H   -2.895233478  -9.996126508  -8.737326665
  P   -1.994799871  -1.684503173  -5.175076785
  O   -2.923266060  -0.392114554  -5.588101851
  C   -2.210186262   0.696598439  -6.210971024
  H   -3.045024600   1.390566196  -6.451510662
  H   -1.740612157   0.358929509  -7.157754673
  C   -1.193558966   1.342459551  -5.256922393
  H   -0.195906304   0.825053128  -5.243946177
  O   -0.894851914   2.667506389  -5.807277926
  C   -1.137272748   3.699272469  -4.844090872
  H   -0.228902570   4.354750433  -4.882811485
  C   -1.717289051   1.561334169  -3.814077773
  H   -2.794841595   1.287901071  -3.697296145
  C   -1.440870699   3.043429390  -3.502914253
  H   -0.555303077   3.105633177  -2.818545298
  H   -2.264498071   3.499314317  -2.925889098
  O   -0.860100967   0.831385483  -2.922301117
  O   -0.738560368  -1.359917416  -4.380477576
  O   -2.888027455  -2.953359869  -4.712411398
  N   -2.303246981   4.487463783  -5.377378036
  C   -2.679662791   4.591258892  -6.711055259
  C   -3.783051775   5.503675948  -6.763795410
  N   -4.075086931   5.956731849  -5.470206496
  C   -3.202333220   5.360815029  -4.657705272
  N   -2.173147609   3.993848853  -7.865007128
  C   -2.793077927   4.362300132  -9.072962686
  N   -3.813631580   5.198767903  -9.169952073
  C   -4.368624179   5.823766814  -8.030739565
  H   -2.393004809   3.907474556 -10.006425692
  N   -5.394753849   6.670608489  -8.201153981
  H   -5.772783753   6.874385475  -9.126981945
  H   -5.819585944   7.149427029  -7.404520393
  H   -3.128332975   5.500181826  -3.582099609
  P   -1.485845177  -0.581521521  -2.339210671
  O   -0.170736139  -1.329863307  -1.761095688
  C   -0.002225389  -2.749522903  -1.970366207
  H   -0.323611181  -3.075332193  -2.978658665
  H    1.107479537  -2.827176571  -1.896053853
  C   -0.718149526  -3.520766944  -0.860339618
  H   -0.835350789  -2.915745095   0.071814541
  O   -2.084232967  -3.782888356  -1.306617021
  C   -2.363705176  -5.183576628  -1.339096984
  H   -3.385685005  -5.278813753  -0.895410390
  C   -0.081767690  -4.903182435  -0.558921866
  H    0.713647114  -5.184566279  -1.305977406
  C   -1.256382087  -5.897011251  -0.563405919
  H   -0.966401503  -6.883020231  -0.982386598
  H   -1.592689417  -6.120002508   0.474120033
  O    0.489679004  -4.756719782   0.751455015
  O   -2.631626354  -1.198953157  -3.109435150
  O   -2.082448148  -0.027743870  -0.919331282
  N   -2.439683840  -5.598202529  -2.786648126
  C   -1.272867622  -5.809957298  -3.569846060
  N   -1.458369662  -6.405647912  -4.841072194
  C   -2.739019474  -6.684897489  -5.430250870
  C   -3.891173920  -6.288848364  -4.608277388
  C   -3.720889303  -5.791153203  -3.358945526
  O   -0.148500663  -5.496828243  -3.207283148
  H   -0.601216308  -6.753770046  -5.299799386
  O   -2.735643494  -7.190381632  -6.534031986
  C   -5.233638169  -6.453246129  -5.228485583
  H   -5.288035054  -7.365437072  -5.846378712
  H   -6.046461547  -6.520587090  -4.491229476
  H   -5.465827844  -5.604539621  -5.893533712
  H   -4.572938742  -5.504343065  -2.725441141
  P    1.543915086  -5.950872297   1.204835405
  O    2.917402944  -5.060392518   1.244286393
  C    4.149854181  -5.700226874   0.830354469
  H    4.865832913  -5.328471667   1.599535166
  H    4.102653392  -6.807993793   0.879705593
  C    4.556464078  -5.233379846  -0.570346753
  H    5.651511447  -5.003752980  -0.590647510
  O    4.424041649  -6.376806861  -1.468623592
  C    3.705757659  -6.007083777  -2.658879912
  H    4.455859119  -5.988088186  -3.489773712
  C    3.727227047  -4.065337712  -1.171724154
  H    3.053837966  -3.583671859  -0.420569129
  C    3.010695306  -4.670574135  -2.385997689
  H    1.920366056  -4.818609716  -2.206027073
  H    3.043194573  -3.987566639  -3.268375668
  O    4.576457416  -2.988416859  -1.522672152
  O    1.536929559  -7.174679392   0.391011657
  O    1.155738180  -6.088194925   2.772001202
  N    2.731386439  -7.113035256  -2.948868908
  C    2.182779757  -7.162515311  -4.290345228
  N    1.326988740  -8.200358122  -4.620301179
  C    0.971374295  -9.140123493  -3.679265466
  C    1.479972755  -9.066093804  -2.343561872
  C    2.362141371  -8.066543224  -2.006635807
  O    2.490038093  -6.292963715  -5.100303337
  N    0.086598852 -10.097164975  -4.098718731
  H   -0.152208197 -10.885680341  -3.514933490
  H   -0.194333378 -10.152700586  -5.077679571
  H    1.020342004  -2.068460142  -4.408889193
  H    2.812987485  -7.983734904  -0.996134173
  H    1.184152643  -9.793553241  -1.587644937
  H    5.174234908  -3.201954970  -2.273008225
  H   -3.129086812  -2.951916839  -3.701355060
  H   -1.448756166   0.106855191  -0.165538052
  H    1.233579875  -5.273050423   3.350107855
  H    1.315812828  -8.367223317  -6.494325064
  H   -1.368528869   3.352454767  -7.791120473
