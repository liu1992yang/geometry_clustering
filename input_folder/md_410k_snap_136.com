%mem=64gb
%nproc=28       
%Chk=snap_136.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_136 

2     1 
  O    1.828580262  -2.295695304  -6.232721493
  C    1.832438679  -3.341287352  -7.196935557
  H    1.911890206  -4.329897350  -6.695922343
  H    2.764379338  -3.157063057  -7.774003464
  C    0.599641691  -3.237272368  -8.107582444
  H    0.550525743  -2.249122701  -8.619981868
  O    0.775883478  -4.224913458  -9.185571309
  C   -0.398241692  -5.009575434  -9.357497225
  H   -0.547539809  -5.140268604 -10.454804056
  C   -0.756760871  -3.594944704  -7.448587003
  H   -0.615626369  -4.214055193  -6.533401949
  C   -1.517539100  -4.365106453  -8.541236524
  H   -2.131916657  -3.665967704  -9.149868126
  H   -2.253312944  -5.085228272  -8.119647089
  O   -1.455514354  -2.371469501  -7.191885715
  N   -0.082602035  -6.374356561  -8.770363164
  C   -0.952185555  -7.485550214  -8.770247585
  C   -0.439449142  -8.404871410  -7.807714902
  N    0.711666083  -7.843760204  -7.244428506
  C    0.905201247  -6.622694427  -7.797983824
  N   -2.068278354  -7.694218395  -9.533938335
  C   -2.734971818  -8.861549648  -9.261640224
  N   -2.320029058  -9.801023537  -8.281324199
  C   -1.144676523  -9.609058190  -7.449966906
  N   -3.892571373  -9.104185279  -9.960588353
  H   -4.157254528  -8.467543724 -10.711634488
  H   -4.376164603  -9.989778509  -9.925624056
  O   -0.894693818 -10.422763203  -6.589745454
  H    1.713753885  -5.915909821  -7.545586386
  H   -2.885917011 -10.638814123  -8.085289289
  P   -1.885724413  -2.179873836  -5.593807802
  O   -2.539574640  -0.672474868  -5.752476040
  C   -1.644087377   0.443331139  -5.633097017
  H   -2.095205076   1.187091532  -6.325638877
  H   -0.614550606   0.213207618  -5.969218602
  C   -1.695526435   0.926761782  -4.172681496
  H   -0.888967345   0.488510175  -3.531717354
  O   -1.363247175   2.351849569  -4.171353636
  C   -2.440301781   3.144443860  -3.659538538
  H   -1.984803595   3.770189906  -2.847163755
  C   -3.115354262   0.784517432  -3.563291908
  H   -3.825567911   0.246025396  -4.242622777
  C   -3.575144676   2.216989986  -3.236440425
  H   -3.772081085   2.287264979  -2.136610173
  H   -4.549027714   2.436104527  -3.712871419
  O   -3.092134238   0.136909488  -2.287079265
  O   -0.672444099  -2.385815976  -4.760688173
  O   -3.184914428  -3.148259723  -5.610678613
  N   -2.841957183   4.064959065  -4.780676862
  C   -2.097976539   4.374395367  -5.911724630
  C   -2.839226119   5.360280253  -6.642971644
  N   -4.025914731   5.653410823  -5.957375916
  C   -4.024054494   4.894022871  -4.861737454
  N   -0.869531484   3.900143806  -6.370844399
  C   -0.416326403   4.436262478  -7.589218260
  N   -1.063772628   5.348579619  -8.296169027
  C   -2.298099583   5.880983565  -7.861984743
  H    0.550988997   4.050872698  -7.981625753
  N   -2.876043227   6.834369325  -8.608804031
  H   -2.437956225   7.183308957  -9.462107593
  H   -3.772643457   7.242728418  -8.340945669
  H   -4.790164685   4.876293649  -4.090013988
  P   -2.689496592  -1.473042529  -2.208262506
  O   -1.201281608  -1.139428838  -1.634162939
  C   -0.246406043  -2.224500284  -1.487647921
  H   -0.102842902  -2.743843576  -2.464312127
  H    0.696180414  -1.696408235  -1.212176510
  C   -0.815254804  -3.093574644  -0.369584253
  H   -0.969523994  -2.529732407   0.583426792
  O   -2.170327882  -3.373476846  -0.875987924
  C   -2.369567365  -4.801790413  -1.057624023
  H   -3.421781462  -4.946657919  -0.705752811
  C   -0.159562425  -4.458432902  -0.089562763
  H    0.682716657  -4.667894827  -0.809244603
  C   -1.300769103  -5.487019407  -0.210024352
  H   -0.945981628  -6.455075547  -0.623347754
  H   -1.698625172  -5.750515872   0.796243875
  O    0.323910783  -4.378749225   1.263470791
  O   -2.829339985  -2.169410470  -3.525974157
  O   -3.838005424  -1.873298704  -1.135591955
  N   -2.289770741  -5.144599089  -2.508775727
  C   -1.045793573  -5.343719277  -3.189483036
  N   -1.121414250  -5.859742426  -4.502753468
  C   -2.338524226  -6.099040697  -5.220034888
  C   -3.565726661  -5.758852882  -4.482622453
  C   -3.512727212  -5.307146760  -3.205846208
  O    0.040387756  -5.135040093  -2.675850635
  H   -0.218467291  -6.119543596  -4.947190682
  O   -2.234869052  -6.538694074  -6.349481274
  C   -4.843413812  -5.926117719  -5.226694320
  H   -5.030181454  -6.985809664  -5.465721810
  H   -5.722039803  -5.560081906  -4.679194184
  H   -4.810680627  -5.385022502  -6.189917188
  H   -4.420178164  -5.049723813  -2.640180767
  P    1.511380820  -5.485576504   1.603801945
  O    2.827898248  -4.789152380   0.910544320
  C    3.512956544  -5.593972735  -0.082290065
  H    4.390727791  -6.047278960   0.427693194
  H    2.879077681  -6.398973810  -0.507281123
  C    3.963250789  -4.571640002  -1.137745093
  H    4.707345297  -3.860166231  -0.712773471
  O    4.704913689  -5.323687611  -2.146098368
  C    4.039242823  -5.288105717  -3.413180801
  H    4.859043641  -5.225424712  -4.167023696
  C    2.808559459  -3.856254362  -1.883290261
  H    1.800311155  -4.238838739  -1.566473787
  C    3.053749538  -4.124083183  -3.375255492
  H    2.091287473  -4.327723466  -3.903188580
  H    3.445020128  -3.233090258  -3.906792779
  O    2.716009756  -2.480251107  -1.570521546
  O    1.222521806  -6.862057798   1.162879851
  O    1.739107318  -5.153122048   3.173254033
  N    3.347669221  -6.633160228  -3.575856317
  C    2.326118560  -6.813094154  -4.579949002
  N    1.696582838  -8.042305366  -4.682622231
  C    2.019467818  -9.081116280  -3.842976169
  C    3.063197610  -8.919029051  -2.864736512
  C    3.706300895  -7.712826078  -2.768814591
  O    2.030735996  -5.898403277  -5.348042084
  N    1.321562219 -10.238328502  -4.024095032
  H    1.472566401 -11.050199770  -3.442591455
  H    0.601496531 -10.311704050  -4.743920499
  H    1.053554120  -2.390082459  -5.599861176
  H    4.538848851  -7.545226643  -2.060646602
  H    3.347806789  -9.750664327  -2.219586102
  H    3.554302649  -1.995307111  -1.737213983
  H   -3.685277758  -3.236101016  -4.717740010
  H   -3.576441076  -1.987954571  -0.172421627
  H    1.545131629  -5.877910144   3.838723279
  H    1.215645594  -8.248023528  -6.415947367
  H   -0.341408583   3.226262139  -5.793007345

