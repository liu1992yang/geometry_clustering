%mem=64gb
%nproc=28       
%Chk=snap_148.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_148 

2     1 
  O    1.156923040  -3.467706094  -4.729130228
  C    1.721261829  -3.572652079  -6.027530887
  H    2.020354654  -4.649151335  -6.089549318
  H    2.636676863  -2.955854100  -6.096444402
  C    0.699807079  -3.186573139  -7.110497704
  H    0.806541195  -2.124430791  -7.428744514
  O    1.038502048  -3.951493584  -8.315082774
  C   -0.076442712  -4.696776970  -8.788311788
  H   -0.069887013  -4.612596626  -9.901832871
  C   -0.777117105  -3.523954076  -6.779836555
  H   -0.881253633  -4.144160541  -5.861121084
  C   -1.319209127  -4.206416661  -8.043235559
  H   -1.896733235  -3.472267450  -8.650084831
  H   -2.057733676  -5.009263505  -7.821189886
  O   -1.512832788  -2.285310853  -6.675722662
  N    0.196999710  -6.143115939  -8.437704829
  C   -0.669984219  -7.224064920  -8.711087396
  C   -0.248147670  -8.301109929  -7.875789595
  N    0.870867592  -7.877200169  -7.149166518
  C    1.112430655  -6.582596702  -7.460307349
  N   -1.703660504  -7.277966787  -9.603643088
  C   -2.410750335  -8.455513351  -9.582540944
  N   -2.111919034  -9.541143199  -8.721384621
  C   -1.024646712  -9.508779367  -7.758613538
  N   -3.484026340  -8.560017799 -10.432296774
  H   -3.671531662  -7.803931569 -11.088111293
  H   -3.983550630  -9.424668112 -10.577948219
  O   -0.905059315 -10.431717360  -6.985845983
  H    1.914049981  -5.948940475  -7.023651763
  H   -2.724040107 -10.368321558  -8.685791404
  P   -1.829944621  -1.730947326  -5.159808312
  O   -2.740832643  -0.429593555  -5.576952822
  C   -2.017690849   0.612774496  -6.269830217
  H   -2.843042129   1.304204297  -6.547419918
  H   -1.563498443   0.217642442  -7.202297509
  C   -0.982600286   1.303761338  -5.366874975
  H    0.038731520   0.847323992  -5.426982637
  O   -0.799956124   2.652541956  -5.912057571
  C   -1.176592934   3.643560690  -4.957867998
  H   -0.425361143   4.464488627  -5.066775925
  C   -1.427429035   1.480019281  -3.889436278
  H   -2.482442679   1.143130849  -3.721672193
  C   -1.227921341   2.976576280  -3.589641319
  H   -0.252522291   3.104341579  -3.050512604
  H   -1.969323774   3.384119646  -2.881734629
  O   -0.478113727   0.791907058  -3.053914158
  O   -0.518616510  -1.377945515  -4.460074959
  O   -2.728186957  -2.932398842  -4.561250279
  N   -2.512639105   4.168322972  -5.407956694
  C   -2.741874499   4.867263125  -6.591427443
  C   -4.138331376   5.183655696  -6.625256813
  N   -4.765969719   4.647127866  -5.492349332
  C   -3.811138700   4.041604294  -4.786695497
  N   -1.888191786   5.259986645  -7.622250481
  C   -2.478072807   5.999496424  -8.668395452
  N   -3.754560761   6.336199124  -8.723902122
  C   -4.663218089   5.956171542  -7.711061796
  H   -1.806414626   6.324431637  -9.495631136
  N   -5.942570987   6.341707006  -7.824678412
  H   -6.265528629   6.897822510  -8.617765454
  H   -6.631690715   6.098441700  -7.111143221
  H   -3.951157990   3.511367955  -3.847657281
  P   -1.079300390  -0.596506188  -2.387224302
  O    0.203368886  -1.469664181  -1.906533234
  C    0.128507177  -2.911724280  -2.010899319
  H   -0.195374618  -3.237773188  -3.022455104
  H    1.206690706  -3.172377687  -1.885384633
  C   -0.738740284  -3.514469492  -0.905230153
  H   -0.864950965  -2.833952629  -0.028350432
  O   -2.086307435  -3.690282738  -1.441995589
  C   -2.500922845  -5.062942766  -1.381735686
  H   -3.561606872  -5.020038472  -1.031409773
  C   -0.263853723  -4.921195799  -0.450876872
  H    0.521001525  -5.357184646  -1.120760393
  C   -1.539331124  -5.782908905  -0.437665938
  H   -1.332625625  -6.835563425  -0.719133442
  H   -1.960706033  -5.833770368   0.590908737
  O    0.232404056  -4.743679042   0.889411255
  O   -2.319813111  -1.152757935  -3.050266306
  O   -1.506112713   0.013376855  -0.926813256
  N   -2.490763186  -5.592755285  -2.790974835
  C   -1.285606444  -6.018215028  -3.423009893
  N   -1.401013453  -6.486908688  -4.754619062
  C   -2.608995717  -6.479258500  -5.522476109
  C   -3.787483587  -5.965937152  -4.815299198
  C   -3.699979277  -5.546986712  -3.527935316
  O   -0.197063153  -6.007710799  -2.876789556
  H   -0.544726278  -6.903203487  -5.152124478
  O   -2.531376695  -6.861554197  -6.676287551
  C   -5.063073434  -5.936908074  -5.582797350
  H   -5.683730865  -6.820905772  -5.361650531
  H   -5.666282131  -5.042982380  -5.366525136
  H   -4.883915822  -5.940954951  -6.671692651
  H   -4.574774014  -5.156136020  -2.988677685
  P    1.437566722  -5.795107169   1.315126286
  O    2.710167663  -4.789839719   1.053509958
  C    4.006545139  -5.409920073   0.860340068
  H    4.638483639  -4.906367668   1.626623153
  H    4.001903579  -6.505225274   1.046924006
  C    4.515840925  -5.109161748  -0.552189269
  H    5.618016243  -4.919763112  -0.521499940
  O    4.394261822  -6.336517215  -1.329919712
  C    3.831560436  -6.067375277  -2.625914955
  H    4.658612384  -6.220701411  -3.364095468
  C    3.772726687  -3.987695463  -1.326397988
  H    2.988033566  -3.489849365  -0.705955192
  C    3.243232571  -4.654807629  -2.603589525
  H    2.133784320  -4.666798113  -2.656785915
  H    3.513491926  -4.081380855  -3.515254453
  O    4.644626450  -2.906627639  -1.609739494
  O    1.487724977  -7.098894597   0.643553321
  O    1.248253439  -5.791981706   2.924693397
  N    2.796910849  -7.126008970  -2.879503506
  C    2.294053155  -7.244443002  -4.234168882
  N    1.419639729  -8.279921181  -4.528732046
  C    0.997245323  -9.138925156  -3.540832599
  C    1.439722846  -8.977242619  -2.188449414
  C    2.335662299  -7.980706872  -1.884043700
  O    2.661580914  -6.444429201  -5.088754391
  N    0.110345394 -10.109971828  -3.927326776
  H   -0.160562761 -10.849711208  -3.295438074
  H   -0.117075043 -10.246871673  -4.911359727
  H    0.904238294  -2.517227343  -4.500015316
  H    2.729334523  -7.825399698  -0.857546476
  H    1.072504100  -9.626967362  -1.393350170
  H    5.394746090  -3.165455729  -2.187513509
  H   -2.875461408  -2.866015009  -3.522066738
  H   -0.793044167   0.166122958  -0.254366488
  H    1.110157390  -4.915262723   3.385241073
  H    1.302475006  -8.413717249  -6.345951650
  H   -0.908513649   4.957782084  -7.639964286

