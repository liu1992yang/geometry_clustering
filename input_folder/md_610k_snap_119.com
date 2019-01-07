%mem=64gb
%nproc=28       
%Chk=snap_119.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_119 

2     1 
  O    3.162969953  -4.525032753  -7.289829026
  C    3.066180832  -3.105540639  -7.220069193
  H    3.770741190  -2.776145925  -8.016720640
  H    3.445895918  -2.751178713  -6.242191262
  C    1.654430718  -2.571373047  -7.494433421
  H    1.667480478  -1.460162597  -7.634824051
  O    1.203928155  -3.064169799  -8.796574953
  C    0.136760264  -3.986162806  -8.645419181
  H   -0.531593983  -3.832231958  -9.528123641
  C    0.563275877  -2.974387100  -6.472537237
  H    0.960661068  -3.544967972  -5.589805297
  C   -0.476072646  -3.781622339  -7.262065272
  H   -1.457241277  -3.257394300  -7.328812072
  H   -0.715390288  -4.736382448  -6.741084958
  O    0.025566524  -1.702907967  -6.045328857
  N    0.741250413  -5.370537117  -8.782513583
  C    0.145949934  -6.587879882  -8.385127808
  C    1.094187410  -7.614577377  -8.657656561
  N    2.261291625  -7.004245828  -9.150193881
  C    2.047886503  -5.659220555  -9.207380436
  N   -1.102623509  -6.768740017  -7.867048579
  C   -1.436966410  -8.077303350  -7.606062667
  N   -0.569012617  -9.161974348  -7.870564216
  C    0.798370185  -8.989310022  -8.339211487
  N   -2.660924146  -8.306443553  -7.039010127
  H   -3.298359785  -7.501050359  -6.909740053
  H   -3.053906704  -9.230069191  -6.936225301
  O    1.485103452  -9.982151666  -8.417746584
  H    2.764948853  -4.887614882  -9.518965011
  H   -0.855043067 -10.124017172  -7.646317271
  P   -0.902946839  -1.720620770  -4.674579454
  O   -1.071011212  -0.066178771  -4.652961941
  C   -0.388243428   0.627308089  -3.579214493
  H    0.335373398   1.305844996  -4.087950929
  H    0.173318015  -0.063241509  -2.913923842
  C   -1.433080406   1.431823213  -2.803218483
  H   -1.210114539   1.436288756  -1.704264893
  O   -1.267223126   2.822232223  -3.214026358
  C   -2.533901317   3.458515081  -3.413273687
  H   -2.542024268   4.346676562  -2.730137215
  C   -2.919975274   1.070901136  -3.074433162
  H   -3.074482204   0.464584344  -3.999350273
  C   -3.635259957   2.427325922  -3.153740888
  H   -4.155813776   2.621893322  -2.182851510
  H   -4.450342090   2.439933936  -3.900576988
  O   -3.468949198   0.386055042  -1.941934214
  O   -0.199567676  -2.321638903  -3.523603247
  O   -2.256621821  -2.275552103  -5.374844295
  N   -2.525889657   3.973732550  -4.816975343
  C   -2.264828073   3.308781487  -6.012277061
  C   -2.318960853   4.295188771  -7.053027788
  N   -2.589332734   5.554247580  -6.496711772
  C   -2.700580233   5.365769634  -5.184932001
  N   -2.011517075   1.974208578  -6.317130402
  C   -1.781110368   1.671330946  -7.669540944
  N   -1.826779946   2.547840150  -8.662035402
  C   -2.097812550   3.910585958  -8.413526353
  H   -1.560721421   0.605119777  -7.905995297
  N   -2.138100452   4.756771287  -9.456057989
  H   -1.968913406   4.438988914 -10.409856410
  H   -2.338520530   5.748986060  -9.319128769
  H   -2.904364645   6.131189145  -4.438433164
  P   -3.068014939  -1.224885437  -1.833181616
  O   -1.931050288  -1.022829624  -0.685828174
  C   -0.988790369  -2.111311241  -0.490291927
  H   -0.449391171  -2.328607062  -1.447009299
  H   -0.274517241  -1.685700047   0.248141684
  C   -1.765846468  -3.297105113   0.063536565
  H   -2.080116134  -3.153459999   1.124975851
  O   -3.008901510  -3.281488031  -0.721633373
  C   -3.358935888  -4.635028048  -1.174666965
  H   -4.419358803  -4.742191764  -0.830645961
  C   -1.153902019  -4.696052999  -0.158044938
  H   -0.390463980  -4.706160155  -0.986775493
  C   -2.375922622  -5.575562600  -0.493914718
  H   -2.085719519  -6.470900219  -1.093421860
  H   -2.815332820  -6.012787947   0.431141754
  O   -0.550946674  -5.051259341   1.088344747
  O   -2.624356537  -1.726626557  -3.177656515
  O   -4.528744025  -1.726715106  -1.335272879
  N   -3.331932303  -4.648709576  -2.660333958
  C   -2.215878534  -5.154251194  -3.411434599
  N   -2.480508345  -5.432176666  -4.775952113
  C   -3.740916805  -5.209581024  -5.429048892
  C   -4.693634866  -4.401049765  -4.651488132
  C   -4.489934964  -4.178636012  -3.330834369
  O   -1.125520875  -5.373238376  -2.926107495
  H   -1.727136935  -5.893571636  -5.319280271
  O   -3.902571837  -5.707227251  -6.524460267
  C   -5.884534309  -3.887805792  -5.380804796
  H   -6.171557210  -4.560122267  -6.210966606
  H   -6.772599540  -3.793250229  -4.738321887
  H   -5.690547774  -2.901193424  -5.829184028
  H   -5.210794713  -3.616748566  -2.712874062
  P    0.352393731  -6.452925253   1.042266820
  O    1.839692736  -5.862570413   0.663685513
  C    2.406547083  -6.313259471  -0.590242211
  H    3.293824753  -6.919124135  -0.300895767
  H    1.716301962  -6.945229542  -1.185986137
  C    2.844246740  -5.036332278  -1.318668931
  H    3.381029391  -4.335501300  -0.632223653
  O    3.850216455  -5.455488501  -2.295111743
  C    3.374289516  -5.275747130  -3.635143150
  H    4.281677228  -4.995259623  -4.222064519
  C    1.742397358  -4.308819503  -2.119127395
  H    0.735717785  -4.800463009  -2.063279846
  C    2.274280008  -4.223631443  -3.560915117
  H    1.452965771  -4.352226876  -4.304835133
  H    2.697574978  -3.212059085  -3.734618226
  O    1.633741092  -3.010530093  -1.532489618
  O   -0.173985268  -7.500561367   0.161967859
  O    0.482278952  -6.719809411   2.630904837
  N    2.914820664  -6.640566515  -4.112787903
  C    1.913126685  -6.851558448  -5.153818494
  N    1.673846972  -8.129806499  -5.601614947
  C    2.311265437  -9.200841978  -5.021821106
  C    3.239139271  -9.018825824  -3.936268045
  C    3.533153847  -7.748059220  -3.529153950
  O    1.359010968  -5.859362294  -5.639199662
  N    2.026950327 -10.445615798  -5.513868433
  H    2.537576867 -11.259632534  -5.200389301
  H    1.516054698 -10.544694090  -6.385549337
  H    2.614940331  -4.974228247  -6.583857541
  H    4.277008115  -7.546176709  -2.736038459
  H    3.708488158  -9.878227424  -3.457802177
  H    1.013567209  -2.451171290  -2.085565379
  H   -3.009996250  -2.548844230  -4.720696921
  H   -4.631498629  -1.962210550  -0.357657289
  H    0.932070124  -6.019555570   3.192335354
  H    3.108266362  -7.510346921  -9.390075388
  H   -1.873826668   1.257450926  -5.572984822
