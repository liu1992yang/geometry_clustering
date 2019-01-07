%mem=64gb
%nproc=28       
%Chk=snap_184.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_184 

2     1 
  O    0.456077895  -1.758445329  -5.903119199
  C    0.508422958  -2.780913559  -6.891831117
  H    0.515977749  -3.776787680  -6.399377598
  H    1.494244918  -2.612367523  -7.377127555
  C   -0.635992259  -2.614120681  -7.904477568
  H   -0.742105069  -1.559904890  -8.244556582
  O   -0.217652289  -3.334177170  -9.115202607
  C   -1.088943611  -4.423235028  -9.385641411
  H   -1.119918444  -4.539824230 -10.495364167
  C   -1.999586223  -3.231303385  -7.489134090
  H   -1.936241802  -3.796355665  -6.528620717
  C   -2.404574240  -4.131835177  -8.665826541
  H   -3.114262784  -3.585737442  -9.334403597
  H   -2.974318925  -5.030897825  -8.356329888
  O   -2.994556124  -2.171514705  -7.428121151
  N   -0.387955782  -5.641270354  -8.804028624
  C   -0.944115538  -6.886624733  -8.447851774
  C    0.079358361  -7.600682219  -7.756167262
  N    1.226873656  -6.798987240  -7.709287797
  C    0.942695412  -5.626667506  -8.324208248
  N   -2.204933984  -7.357688370  -8.684885528
  C   -2.456967768  -8.612370749  -8.169059571
  N   -1.511186545  -9.357456506  -7.432652038
  C   -0.157806978  -8.903439415  -7.185339329
  N   -3.711274477  -9.132838872  -8.350852108
  H   -4.377543339  -8.646039158  -8.947590370
  H   -3.944352584 -10.081756238  -8.084068687
  O    0.566441747  -9.610041858  -6.524065514
  H    1.626896680  -4.783000782  -8.471699605
  H   -1.789062342 -10.229050204  -6.911156379
  P   -3.324944740  -1.663644663  -5.895241710
  O   -3.580275473  -0.051534548  -5.992317059
  C   -2.470997289   0.825915454  -6.261222481
  H   -2.950801871   1.822609661  -6.133297024
  H   -2.167751809   0.709725015  -7.321712656
  C   -1.282829721   0.638623205  -5.309757583
  H   -0.635980791  -0.241520692  -5.612318982
  O   -0.407054616   1.791238357  -5.517406501
  C    0.142773586   2.234512368  -4.259179676
  H    1.224997413   1.929754506  -4.265757790
  C   -1.618634991   0.595178666  -3.798944031
  H   -2.699028688   0.773355762  -3.583623058
  C   -0.694719110   1.628510692  -3.133370671
  H   -0.043797312   1.113619722  -2.377346538
  H   -1.276454420   2.371606182  -2.554670305
  O   -1.168674350  -0.679960311  -3.288720660
  O   -2.070296800  -2.049927271  -5.111842037
  O   -4.767425775  -2.302511651  -5.617416132
  N    0.079299575   3.728755944  -4.306902646
  C   -0.044601801   4.530321913  -5.435666390
  C    0.107725325   5.888214593  -5.000852119
  N    0.329386341   5.912246029  -3.616860302
  C    0.320339424   4.643845403  -3.210738296
  N   -0.269103880   4.217546067  -6.775309902
  C   -0.323495624   5.306658849  -7.664905426
  N   -0.185709456   6.573067753  -7.309671503
  C    0.043047154   6.944995104  -5.965831511
  H   -0.500920285   5.074154352  -8.738745806
  N    0.188462165   8.248660410  -5.685780547
  H    0.132583510   8.960761774  -6.415695680
  H    0.362384865   8.568558461  -4.731849451
  H    0.479987586   4.288377177  -2.195102311
  P   -2.353519978  -1.789580699  -2.960314076
  O   -1.555310287  -3.219353388  -3.100369028
  C   -2.390526913  -4.393880474  -2.941134292
  H   -3.419736995  -4.259407082  -3.323948138
  H   -1.872167193  -5.150293348  -3.573910436
  C   -2.362145779  -4.772121827  -1.451591329
  H   -2.860077330  -4.021530820  -0.789811890
  O   -3.171902670  -5.963422866  -1.303793520
  C   -2.395274618  -7.063472715  -0.798855453
  H   -3.042562253  -7.517030777  -0.005923997
  C   -0.934405289  -5.118976514  -0.959293442
  H   -0.182181232  -5.114119924  -1.796524337
  C   -1.072043600  -6.492722735  -0.282971005
  H   -0.193690765  -7.141616837  -0.498248984
  H   -1.101611329  -6.410245696   0.824991341
  O   -0.593741564  -4.044452646  -0.050515774
  O   -3.700073873  -1.538093189  -3.583258087
  O   -2.384001793  -1.666059569  -1.331939985
  N   -2.256625070  -8.057651728  -1.915548785
  C   -1.275597602  -7.908984585  -2.938278294
  N   -1.253864894  -8.908249715  -3.941499967
  C   -2.240415133  -9.939820834  -4.088768646
  C   -3.249022687  -9.981031263  -3.023123459
  C   -3.235646381  -9.076711316  -2.012548650
  O   -0.482033676  -6.983096652  -2.971043145
  H   -0.424311082  -8.905523738  -4.559696986
  O   -2.182708268 -10.640033747  -5.083522246
  C   -4.270036392 -11.060507984  -3.112200739
  H   -4.177098743 -11.781235874  -2.283255212
  H   -5.296523348 -10.665327645  -3.096263583
  H   -4.168796067 -11.649698329  -4.040961168
  H   -3.992485841  -9.095496980  -1.214553174
  P    0.906352539  -4.174080674   0.653029726
  O    1.417862937  -2.637708204   0.395879377
  C    2.793156401  -2.489494382  -0.059981164
  H    3.139502896  -1.617734948   0.536978373
  H    3.425214131  -3.373883809   0.177555626
  C    2.824533741  -2.195265716  -1.562819451
  H    3.549664842  -1.379248406  -1.806470790
  O    3.373750681  -3.378558605  -2.219777669
  C    2.568080984  -3.708375589  -3.375361129
  H    3.094980575  -3.294897046  -4.273169468
  C    1.441623068  -1.913119805  -2.194859544
  H    0.645799495  -1.727552687  -1.442121160
  C    1.181740559  -3.104170999  -3.118560714
  H    0.485686530  -3.846218682  -2.667290333
  H    0.681775109  -2.807307473  -4.067117419
  O    1.596897058  -0.685588460  -2.932843773
  O    1.751427980  -5.280595309   0.192293116
  O    0.494566639  -4.153892769   2.220647525
  N    2.529562761  -5.196545306  -3.489486054
  C    2.234362995  -5.752628014  -4.800964275
  N    2.173758549  -7.122859145  -4.951218988
  C    2.354938830  -7.939099625  -3.862804055
  C    2.677998625  -7.402224473  -2.570337437
  C    2.754678226  -6.042318417  -2.406504148
  O    2.045480410  -4.982883817  -5.742544139
  N    2.142491511  -9.279230961  -4.071826868
  H    2.417524664  -9.963436426  -3.383265727
  H    1.985189219  -9.625244692  -5.019907230
  H   -0.314035703  -1.915724098  -5.265914840
  H    2.994396884  -5.569003773  -1.431558945
  H    2.849913020  -8.060426805  -1.720533272
  H    0.924769336  -0.668011599  -3.650909530
  H   -5.159285470  -2.116848908  -4.671421675
  H   -1.655379728  -2.118009066  -0.808592015
  H    0.040685196  -3.346636405   2.591663125
  H    2.022201567  -7.004083193  -7.072718578
  H   -0.358877826   3.229738998  -7.055750584
