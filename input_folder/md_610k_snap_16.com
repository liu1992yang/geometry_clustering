%mem=64gb
%nproc=28       
%Chk=snap_16.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_16 

2     1 
  O   -0.356799437  -6.321631192  -7.462283110
  C    0.678135004  -5.449682837  -6.998512667
  H    1.609574083  -5.901879601  -7.408963272
  H    0.724233244  -5.485577231  -5.888967665
  C    0.516303420  -4.008685060  -7.498657363
  H    1.066335889  -3.298040636  -6.825521488
  O    1.264794150  -3.872579262  -8.746050100
  C    0.404953695  -3.707266616  -9.858504928
  H    0.906664883  -2.954557844 -10.522432182
  C   -0.938701554  -3.570843869  -7.805717726
  H   -1.718315346  -4.286226689  -7.444400511
  C   -0.984226632  -3.343745213  -9.328478369
  H   -1.211475435  -2.264713667  -9.520877150
  H   -1.812594629  -3.889074291  -9.814968803
  O   -1.230325519  -2.253576567  -7.279898314
  N    0.400263553  -5.018240561 -10.612744119
  C    1.078186031  -5.274224748 -11.828669764
  C    0.956674157  -6.672404330 -12.076307576
  N    0.236356879  -7.240635309 -11.010934277
  C   -0.079806439  -6.250840920 -10.138923629
  N    1.713161291  -4.377069116 -12.635242927
  C    2.288522175  -4.924722625 -13.764360428
  N    2.232900057  -6.300499791 -14.073434315
  C    1.547964012  -7.289579252 -13.242806536
  N    2.934214685  -4.071809807 -14.617501944
  H    2.993568698  -3.080901714 -14.390805772
  H    3.405354331  -4.382513942 -15.456297932
  O    1.557769473  -8.435344866 -13.603101592
  H   -0.603453771  -6.358963969  -9.143610424
  H    2.691609518  -6.670927511 -14.921125201
  P   -1.308660777  -2.063598497  -5.655091873
  O   -1.486364665  -0.417100595  -5.674228810
  C   -0.640169888   0.328679841  -4.769611535
  H   -0.094445563   1.043421804  -5.427015797
  H    0.095400105  -0.315893394  -4.236337258
  C   -1.531196838   1.082922028  -3.779107792
  H   -1.028296410   1.179107670  -2.778250964
  O   -1.611544412   2.473896502  -4.233803124
  C   -2.953479642   2.882299806  -4.435480904
  H   -3.031917116   3.886961461  -3.959390177
  C   -2.990268298   0.573201008  -3.635950824
  H   -3.256510297  -0.266743635  -4.319982635
  C   -3.880334212   1.805612404  -3.870452443
  H   -4.318165426   2.118338840  -2.888450969
  H   -4.756180768   1.576712975  -4.505904748
  O   -3.232497115   0.222516976  -2.263791914
  O   -0.191838028  -2.621004885  -4.862173624
  O   -2.779951864  -2.528942582  -5.229599203
  N   -3.143929425   3.036753332  -5.921333684
  C   -3.443537241   4.199892424  -6.623856292
  C   -3.541348502   3.832563072  -8.007000473
  N   -3.266931121   2.463541734  -8.149419046
  C   -3.024616525   2.001868481  -6.922387021
  N   -3.651216594   5.519895105  -6.218210647
  C   -3.959640156   6.445881283  -7.235755477
  N   -4.077714817   6.142625409  -8.516052812
  C   -3.880816378   4.824271630  -8.979858102
  H   -4.119427517   7.503146683  -6.923752497
  N   -4.030090844   4.590688472 -10.293923331
  H   -4.281995160   5.339360220 -10.940429822
  H   -3.898974662   3.655624888 -10.677467678
  H   -2.746754213   0.975800758  -6.649809472
  P   -2.775864782  -1.295087257  -1.794351154
  O   -1.163846723  -1.006983408  -1.929020784
  C   -0.215082799  -2.051111073  -1.605999602
  H   -0.119592498  -2.713754029  -2.499265091
  H    0.734250923  -1.490736613  -1.471047558
  C   -0.650414233  -2.772269409  -0.331970304
  H   -0.653917530  -2.122571340   0.576978725
  O   -2.067694533  -3.091351505  -0.564595964
  C   -2.255565869  -4.543062383  -0.704995907
  H   -3.246105352  -4.713947397  -0.217940217
  C    0.061853500  -4.117943910  -0.056419979
  H    0.850175070  -4.349789782  -0.818391860
  C   -1.063458633  -5.158563587  -0.005323205
  H   -0.756570982  -6.158412959  -0.420139386
  H   -1.293316839  -5.432430949   1.058320948
  O    0.622519474  -3.983489028   1.267752934
  O   -3.333811447  -2.362100938  -2.688759948
  O   -3.314356072  -1.124523997  -0.281598143
  N   -2.359489223  -4.871546941  -2.159208569
  C   -1.207681333  -5.132627770  -2.951464242
  N   -1.437801622  -5.479464090  -4.303652321
  C   -2.715294988  -5.470390653  -4.951166577
  C   -3.832564284  -5.106938741  -4.088346631
  C   -3.636934845  -4.808884562  -2.774654004
  O   -0.057985531  -5.085474790  -2.543025717
  H   -0.592436801  -5.695907860  -4.859234476
  O   -2.699264061  -5.734398940  -6.148626235
  C   -5.178016963  -5.022314337  -4.722243326
  H   -5.168044221  -5.397980806  -5.759670817
  H   -5.931126696  -5.613558779  -4.180741808
  H   -5.529301794  -3.978427049  -4.767969046
  H   -4.467577493  -4.495644012  -2.123850635
  P    2.083679762  -4.686737778   1.531358901
  O    2.154475052  -5.701985296   0.250066872
  C    3.457750742  -6.261718743  -0.090823507
  H    4.069239034  -5.445298373  -0.524455120
  H    3.937003362  -6.689385909   0.815452221
  C    3.200243454  -7.377289234  -1.114469422
  H    3.993586257  -7.398937688  -1.897437928
  O    3.359757497  -8.642351415  -0.409897882
  C    2.087414207  -9.333330910  -0.343383999
  H    2.352093859 -10.413317842  -0.292705657
  C    1.782532283  -7.415719521  -1.753163008
  H    1.067399450  -6.716059876  -1.263667488
  C    1.344398633  -8.883767985  -1.599965903
  H    0.241637688  -8.999414315  -1.538028045
  H    1.661055829  -9.478436467  -2.480648100
  O    1.896685396  -7.075600058  -3.138305671
  O    2.323007642  -5.144772938   2.900396622
  O    3.114033631  -3.507705016   1.010058645
  N    1.468127370  -8.902593572   0.957843091
  C    0.266348659  -8.061411482   1.051031829
  N    0.069558959  -7.318203656   2.201949307
  C    0.891461189  -7.463798295   3.279479823
  C    2.007622536  -8.382297568   3.251891038
  C    2.287007908  -9.036757213   2.088282991
  O   -0.496932181  -8.012610084   0.096547046
  N    0.581578371  -6.742934139   4.414318352
  H    1.324972676  -6.534779257   5.070446422
  H   -0.111900773  -6.006133524   4.331495683
  H   -1.173631327  -6.268996356  -6.863056038
  H    3.169288823  -9.687178439   1.988169057
  H    2.619319418  -8.523965750   4.140725467
  H    1.331288659  -6.282851968  -3.301908399
  H   -2.966403115  -2.631482800  -4.157568208
  H   -3.294280124  -1.939868493   0.325386812
  H    3.646693108  -3.017684750   1.687534935
  H    0.012042403  -8.230704985 -10.940026666
  H   -3.535887255   5.806344406  -5.243272144
