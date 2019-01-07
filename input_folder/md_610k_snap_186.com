%mem=64gb
%nproc=28       
%Chk=snap_186.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_186 

2     1 
  O    2.862567772  -4.143907924  -4.804683897
  C    3.003958239  -2.999604981  -5.656294365
  H    3.988774937  -3.175712163  -6.143949297
  H    3.065446581  -2.100276788  -5.010826907
  C    1.898246430  -2.850331471  -6.705597437
  H    1.876655024  -1.817184630  -7.135618386
  O    2.224564836  -3.676369594  -7.863843672
  C    1.280196710  -4.722205609  -8.042732699
  H    1.064078203  -4.750135270  -9.142601158
  C    0.498477354  -3.289556981  -6.221581741
  H    0.501930202  -3.573485574  -5.135839124
  C    0.089491766  -4.466789116  -7.118190202
  H   -0.826981999  -4.214889697  -7.695831552
  H   -0.196396627  -5.346418948  -6.498622137
  O   -0.369979041  -2.160500446  -6.480174808
  N    1.993677494  -6.008440257  -7.700770597
  C    2.265497721  -7.054572472  -8.603392883
  C    3.018940500  -8.032077306  -7.887166814
  N    3.212100294  -7.549885613  -6.577207704
  C    2.606854050  -6.344138501  -6.476661410
  N    1.893261632  -7.144008852  -9.917141636
  C    2.323946237  -8.275296497 -10.563012578
  N    3.073425526  -9.295166579  -9.924893316
  C    3.465795770  -9.239396334  -8.526596574
  N    1.999516660  -8.416453065 -11.886333812
  H    1.478297298  -7.678476446 -12.356760915
  H    2.299561869  -9.199077389 -12.448534388
  O    4.094470237 -10.183460045  -8.100533878
  H    2.606764051  -5.641391750  -5.579627806
  H    3.379207547 -10.131292848 -10.444043760
  P   -1.453473414  -1.867249718  -5.269110852
  O   -2.442598462  -0.753216149  -5.964314709
  C   -1.845336337   0.525991093  -6.268442460
  H   -2.655851850   1.031638247  -6.835127351
  H   -0.977413971   0.403619874  -6.948821709
  C   -1.479001792   1.297121469  -4.992292331
  H   -0.443461708   1.077710430  -4.620937913
  O   -1.428365774   2.702220371  -5.367804331
  C   -2.349256254   3.482224059  -4.604472035
  H   -1.764114005   4.383104764  -4.277907859
  C   -2.514413491   1.168372445  -3.844032029
  H   -3.401597294   0.557046415  -4.130419039
  C   -2.881453965   2.615483525  -3.468686401
  H   -2.360809775   2.867647593  -2.506837623
  H   -3.947097745   2.752975298  -3.218100000
  O   -1.832086613   0.631482155  -2.706437075
  O   -0.678324009  -1.441547145  -4.043433741
  O   -2.260316453  -3.257445392  -5.310326363
  N   -3.387965954   3.970904590  -5.568477626
  C   -4.625031131   3.457283482  -5.950818750
  C   -5.142001934   4.338121289  -6.961007402
  N   -4.209871544   5.351608777  -7.228264054
  C   -3.177512098   5.126192661  -6.421071560
  N   -5.366511952   2.352940416  -5.543802841
  C   -6.648103114   2.210717188  -6.104104602
  N   -7.166936856   3.005784263  -7.027626351
  C   -6.449085963   4.122070645  -7.501129422
  H   -7.253055813   1.351751606  -5.732800084
  N   -7.036887805   4.916836856  -8.411766686
  H   -7.978353218   4.727443628  -8.753673984
  H   -6.557359645   5.739521422  -8.776550458
  H   -2.260252341   5.710369748  -6.366879157
  P   -2.135738870  -0.960597711  -2.363486639
  O   -0.818978147  -1.361768626  -1.502782423
  C   -0.334135635  -2.711279604  -1.664862027
  H    0.124313644  -2.822023163  -2.669006046
  H    0.476367954  -2.746723270  -0.903090433
  C   -1.456447772  -3.707973146  -1.354012443
  H   -1.264015264  -4.248880580  -0.396590618
  O   -2.658483945  -2.900678363  -1.150905143
  C   -3.846387150  -3.763919027  -1.364598144
  H   -4.186022855  -4.035680787  -0.333444474
  C   -1.874574256  -4.701515768  -2.469006946
  H   -1.704042850  -4.339403774  -3.511884523
  C   -3.360198068  -4.963326906  -2.181504224
  H   -3.941225305  -5.139027766  -3.117321092
  H   -3.474368398  -5.916738157  -1.615995251
  O   -1.204534351  -5.969684017  -2.252506249
  O   -2.882224579  -1.657766354  -3.496610718
  O   -3.173052622  -0.630522267  -1.117789443
  N   -4.879636093  -2.884135731  -1.973994716
  C   -4.928936632  -2.691305755  -3.391423593
  N   -5.439661623  -1.451357001  -3.869647816
  C   -5.784651015  -0.367190243  -3.018349226
  C   -5.933879950  -0.708880625  -1.603296631
  C   -5.427461130  -1.875979139  -1.126683271
  O   -4.675412599  -3.584695593  -4.190221405
  H   -5.335501515  -1.287373248  -4.881157166
  O   -5.912115316   0.736962702  -3.543335664
  C   -6.582653740   0.296514040  -0.718866522
  H   -7.469705664  -0.114281562  -0.211466363
  H   -5.889535751   0.661584634   0.056525642
  H   -6.922731017   1.183200997  -1.279488934
  H   -5.433615976  -2.115136460  -0.054448133
  P    0.316753441  -6.033150822  -2.847767461
  O    1.106230554  -5.159565140  -1.687450235
  C    2.530808901  -5.356049920  -1.551833021
  H    3.021514300  -5.668180969  -2.500863421
  H    2.887977715  -4.332135325  -1.297772033
  C    2.770530638  -6.323871763  -0.392213005
  H    2.026083420  -6.151296612   0.423306121
  O    2.486353467  -7.700014838  -0.798173377
  C    3.658980858  -8.541901690  -0.680300288
  H    3.381178884  -9.269143078   0.125195284
  C    4.238231573  -6.298863929   0.116610093
  H    4.822908846  -5.419922149  -0.245311550
  C    4.843356039  -7.639436455  -0.331184592
  H    5.521490762  -7.487120575  -1.198825612
  H    5.492066627  -8.073912880   0.455974760
  O    4.273553785  -6.113175599   1.517780215
  O    0.543307608  -5.378580993  -4.167225348
  O    0.672678389  -7.592743045  -2.712823298
  N    3.813382361  -9.277830002  -1.968086635
  C    3.870531101  -8.591033291  -3.259017986
  N    4.087126637  -9.340398105  -4.397137488
  C    4.297357132 -10.695271567  -4.324470362
  C    4.235102268 -11.385865063  -3.053477853
  C    3.988755889 -10.667848060  -1.921507478
  O    3.685939013  -7.383776074  -3.287293297
  N    4.554576202 -11.350103081  -5.486009391
  H    4.726465271 -12.345167880  -5.518719753
  H    4.572020781 -10.845707145  -6.378405568
  H    1.945622868  -4.198659594  -4.389210356
  H    3.922509087 -11.155946589  -0.937547862
  H    4.379676970 -12.466930970  -3.010178605
  H    3.875655782  -6.862676302   2.013678268
  H   -3.234713770  -3.258009720  -4.947925724
  H   -2.808266863  -0.761266287  -0.198887872
  H    1.611901844  -7.812269930  -2.309387430
  H    3.704852732  -8.079575314  -5.822693468
  H   -5.070670616   1.714373262  -4.769197482

