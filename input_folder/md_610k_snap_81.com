%mem=64gb
%nproc=28       
%Chk=snap_81.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_81 

2     1 
  O    3.427587513  -5.506877249  -7.060587517
  C    3.511689134  -4.145387321  -7.477031284
  H    4.293521777  -4.187206981  -8.270935214
  H    3.878853455  -3.525407632  -6.635610213
  C    2.210623394  -3.568348568  -8.039771818
  H    2.364810222  -2.509592108  -8.366522785
  O    1.906172708  -4.291667368  -9.285863190
  C    0.554244264  -4.721676003  -9.305582437
  H    0.191517043  -4.605686934 -10.354865694
  C    0.933732460  -3.685585375  -7.169964556
  H    1.005246945  -4.499700248  -6.405737497
  C   -0.178094767  -3.960185480  -8.201638642
  H   -0.617663996  -3.017481239  -8.589179914
  H   -1.042322707  -4.508570130  -7.764444611
  O    0.790393026  -2.417588116  -6.506882316
  N    0.555600229  -6.207277527  -8.986588563
  C   -0.602121498  -7.001567249  -8.830572499
  C   -0.176567759  -8.244217172  -8.279991257
  N    1.218333076  -8.185820680  -8.117711035
  C    1.644217921  -6.957391055  -8.516119849
  N   -1.889316791  -6.670485636  -9.146606877
  C   -2.812854922  -7.651171342  -8.869372279
  N   -2.478170178  -8.899030554  -8.290997938
  C   -1.129853814  -9.255392485  -7.891372438
  N   -4.112077606  -7.430080222  -9.265234831
  H   -4.352708236  -6.523879192  -9.660786172
  H   -4.878674176  -7.974277965  -8.895563224
  O   -0.968514170 -10.310699216  -7.320807145
  H    2.684620109  -6.557467179  -8.480740771
  H   -3.220938870  -9.550296238  -8.000374943
  P   -0.387801725  -2.301964107  -5.337715921
  O   -1.309268478  -1.108735884  -6.015436246
  C   -0.972651191   0.241099183  -5.660796353
  H   -1.527292917   0.824400372  -6.427214774
  H    0.113253622   0.436170482  -5.776239438
  C   -1.442552294   0.538148347  -4.231764755
  H   -0.936116793  -0.132191876  -3.483979625
  O   -0.926649357   1.871248334  -3.890305215
  C   -1.844146461   2.545803562  -3.005590008
  H   -1.272185886   2.745266935  -2.062458145
  C   -2.976617811   0.649774370  -4.026996904
  H   -3.504879548   0.997357987  -4.950831539
  C   -3.088767956   1.673361521  -2.871679662
  H   -3.101419602   1.144321820  -1.887170369
  H   -4.050279645   2.216008916  -2.905261549
  O   -3.699893870  -0.524338948  -3.718179294
  O    0.239761948  -1.967355607  -4.033843439
  O   -1.150254751  -3.682890719  -5.709685585
  N   -2.134797071   3.863221477  -3.662194878
  C   -1.421757296   4.467077336  -4.690200466
  C   -2.035798580   5.739807187  -4.931184503
  N   -3.110625408   5.913886265  -4.048640577
  C   -3.164938194   4.813164358  -3.300262650
  N   -0.312074535   4.045226184  -5.421412141
  C    0.171461951   4.948011129  -6.386793195
  N   -0.366628538   6.127617646  -6.648656078
  C   -1.495141622   6.599647902  -5.941010956
  H    1.066685595   4.636022577  -6.969849991
  N   -1.985703921   7.807943922  -6.253933168
  H   -1.560233702   8.386400304  -6.979561327
  H   -2.795186090   8.191848710  -5.762394247
  H   -3.866316571   4.617332698  -2.492990569
  P   -3.055968301  -1.889796491  -3.023436403
  O   -2.340283171  -1.085654526  -1.792347965
  C   -1.224543888  -1.796192488  -1.177533011
  H   -0.468339027  -2.057671704  -1.951248315
  H   -0.791817993  -1.038605327  -0.488447860
  C   -1.826549265  -2.987538689  -0.444017020
  H   -2.285394012  -2.699998917   0.531551411
  O   -2.955056618  -3.402634382  -1.295222526
  C   -3.089139425  -4.863802125  -1.282895573
  H   -4.129875483  -5.014379059  -0.897073831
  C   -0.949277440  -4.255362880  -0.319291984
  H   -0.204028442  -4.340170431  -1.157591355
  C   -1.987812848  -5.393022297  -0.369238290
  H   -1.539890061  -6.353134654  -0.690593679
  H   -2.363903793  -5.594722644   0.661740671
  O   -0.333032813  -4.351536394   0.955971105
  O   -2.161548417  -2.670710072  -3.924679036
  O   -4.526007695  -2.535656356  -2.739563867
  N   -3.030909664  -5.346888840  -2.691776738
  C   -1.840258771  -5.871741443  -3.283830764
  N   -1.999753034  -6.487090778  -4.546306410
  C   -3.226282433  -6.551683433  -5.287251348
  C   -4.336966714  -5.798957812  -4.690930272
  C   -4.223039840  -5.251408430  -3.456883773
  O   -0.742506997  -5.808889689  -2.755716438
  H   -1.153306294  -6.928129394  -4.965849780
  O   -3.224485985  -7.190948839  -6.320508684
  C   -5.578955798  -5.685706961  -5.504935348
  H   -6.001444780  -4.669898215  -5.484436800
  H   -5.396644923  -5.932212160  -6.563081457
  H   -6.358207780  -6.378500353  -5.146662751
  H   -5.051371944  -4.693028715  -2.987417047
  P    1.056338707  -3.480855526   1.259567812
  O    2.298982260  -4.493035011   0.901371098
  C    2.238704385  -5.468504715  -0.147555630
  H    2.882257132  -6.288345827   0.257924175
  H    1.228244748  -5.876912899  -0.312054409
  C    2.897420465  -4.859672822  -1.391857828
  H    3.760517648  -4.205210693  -1.101913184
  O    3.503067412  -5.971673294  -2.122127703
  C    3.127341002  -5.962308623  -3.504169524
  H    4.079796678  -6.128999364  -4.065936269
  C    1.966058938  -4.142985544  -2.392799766
  H    0.872215604  -4.325998861  -2.211553689
  C    2.411739399  -4.637691238  -3.772700292
  H    1.548730952  -4.728067831  -4.463497186
  H    3.102353942  -3.901619332  -4.234514529
  O    2.235980522  -2.740146303  -2.243469913
  O    1.111644112  -2.912850209   2.591183846
  O    0.974116589  -2.459591087  -0.020852632
  N    2.245389824  -7.175317664  -3.721417490
  C    1.360233140  -7.278371125  -4.866223087
  N    0.634494470  -8.424594905  -5.067243746
  C    0.702246834  -9.459836666  -4.160677974
  C    1.534229229  -9.354689440  -2.989453775
  C    2.286218297  -8.224937812  -2.803379083
  O    1.288530222  -6.328577593  -5.654533434
  N   -0.063597380 -10.550573989  -4.438335031
  H   -0.027800926 -11.386628745  -3.874052633
  H   -0.580959178 -10.607504613  -5.322051918
  H    2.762729923  -5.633146574  -6.325682401
  H    2.958297121  -8.099614387  -1.934656368
  H    1.571720468 -10.169025301  -2.266174924
  H    1.662380978  -2.218625792  -2.902090922
  H   -1.724968415  -4.121551661  -5.010244255
  H   -5.234736062  -1.902988293  -2.406120859
  H    1.762332037  -2.377898078  -0.688145741
  H    1.758432743  -8.914148238  -7.661775878
  H    0.119839741   3.134969635  -5.199858644
