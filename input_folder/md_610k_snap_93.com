%mem=64gb
%nproc=28       
%Chk=snap_93.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_93 

2     1 
  O    2.553029831  -4.498213849  -6.909532505
  C    2.974451873  -3.139236925  -6.827948005
  H    3.900913947  -3.121413155  -7.441406716
  H    3.218874099  -2.867353317  -5.780739355
  C    1.938704320  -2.165700168  -7.403323901
  H    2.187102430  -1.110205880  -7.142601241
  O    2.064140061  -2.173111672  -8.868150947
  C    0.891417048  -2.715653203  -9.462395274
  H    0.812207235  -2.252008618 -10.474715367
  C    0.454450336  -2.511671824  -7.102169176
  H    0.326379320  -3.502006828  -6.600962891
  C   -0.242378375  -2.451439472  -8.476821242
  H   -0.679030796  -1.439469910  -8.638965849
  H   -1.109409843  -3.142168409  -8.554639238
  O   -0.158891788  -1.468417836  -6.338194755
  N    1.173750817  -4.193439253  -9.647022528
  C    0.486880739  -5.306203219  -9.102525432
  C    1.351011613  -6.432424048  -9.252382223
  N    2.543568288  -5.991981901  -9.842703350
  C    2.444365486  -4.655139507 -10.054948104
  N   -0.754600526  -5.340357031  -8.544055030
  C   -1.147537910  -6.569305261  -8.056162779
  N   -0.352100298  -7.725680541  -8.166806137
  C    0.982689046  -7.737596584  -8.743294624
  N   -2.415135191  -6.679646161  -7.525511050
  H   -2.986295170  -5.836837669  -7.497102873
  H   -2.580354802  -7.302475415  -6.729274099
  O    1.607224395  -8.766341026  -8.729758609
  H    3.210530680  -3.995885259 -10.475088415
  H   -0.671934577  -8.605000468  -7.732361679
  P    0.047232411  -1.477041945  -4.666581603
  O    0.199602911   0.168426758  -4.738227893
  C    0.464908103   0.969699133  -3.565598977
  H    1.133814151   1.767178614  -3.971937445
  H    1.009606965   0.390803218  -2.788699609
  C   -0.826633822   1.586893929  -3.024817643
  H   -0.746071103   1.756111683  -1.919383785
  O   -0.902192252   2.930830512  -3.601310425
  C   -2.259050126   3.294165553  -3.880457160
  H   -2.449392871   4.246305400  -3.321933737
  C   -2.183179612   0.920364759  -3.384648262
  H   -2.173507955   0.330677671  -4.335163069
  C   -3.150025498   2.108117211  -3.490673280
  H   -3.652250796   2.274803840  -2.506100133
  H   -3.988246576   1.923014734  -4.188690345
  O   -2.663738590   0.091145199  -2.309490534
  O    1.113897743  -2.394178477  -4.254756458
  O   -1.615958126  -1.615907301  -4.444383647
  N   -2.320733469   3.606396721  -5.342702638
  C   -1.886604328   2.856027201  -6.432241189
  C   -2.171389756   3.626588943  -7.608179546
  N   -2.756720269   4.847006944  -7.239090836
  C   -2.835420921   4.837347426  -5.910922601
  N   -1.308432803   1.595610824  -6.539036032
  C   -1.022614761   1.140483124  -7.831474324
  N   -1.276548539   1.809940064  -8.949023836
  C   -1.847147589   3.099754172  -8.899301612
  H   -0.565920972   0.115949960  -7.901393919
  N   -2.058213119   3.755005960 -10.053769530
  H   -1.811277079   3.348118278 -10.953726037
  H   -2.466878718   4.690976173 -10.057320201
  H   -3.235998604   5.630881838  -5.282610745
  P   -1.862894648  -1.326888732  -2.119486378
  O   -1.903864281  -1.468604640  -0.501534466
  C   -1.104665996  -2.598534030  -0.036237298
  H   -0.423868717  -3.003467339  -0.816293711
  H   -0.469146928  -2.148723254   0.769736276
  C   -2.033694458  -3.652471250   0.558836931
  H   -2.439430236  -3.329740732   1.546881284
  O   -3.223952939  -3.831685672  -0.293751690
  C   -3.656225795  -5.215166927  -0.241844460
  H   -4.688281377  -5.173816133   0.187169566
  C   -1.402034977  -5.073434043   0.599974244
  H   -0.771539766  -5.261492960  -0.313622543
  C   -2.637591471  -5.989970196   0.596083494
  H   -2.416605028  -7.004565479   0.222199084
  H   -2.987610310  -6.152895371   1.642574484
  O   -0.676481229  -5.292033682   1.801090667
  O   -0.411823727  -1.290819676  -2.609409803
  O   -2.821316377  -2.514403791  -2.661479479
  N   -3.746124012  -5.689128576  -1.665145467
  C   -2.589708946  -5.824950443  -2.491136766
  N   -2.796778453  -6.432884992  -3.745579310
  C   -4.013345116  -7.047531410  -4.185511059
  C   -5.158953228  -6.831536203  -3.294583166
  C   -5.000706890  -6.178927666  -2.114304475
  O   -1.477519324  -5.432855176  -2.169841681
  H   -1.966947380  -6.526898927  -4.372954958
  O   -3.948641513  -7.664748556  -5.236639814
  C   -6.477205524  -7.361219660  -3.737287017
  H   -6.840847345  -8.166884190  -3.076961714
  H   -7.256268904  -6.582652941  -3.759988745
  H   -6.433124769  -7.792477682  -4.753764186
  H   -5.853446012  -5.998878115  -1.440406226
  P    0.979654057  -5.143182711   1.712884935
  O    1.543670768  -6.415405343   0.831983816
  C    1.283974836  -6.614488867  -0.560899255
  H    1.437320977  -7.714806818  -0.669725201
  H    0.235413583  -6.364560355  -0.832899441
  C    2.320520371  -5.857064058  -1.399133687
  H    3.187560146  -5.519047508  -0.779419642
  O    2.911973795  -6.829164358  -2.321502473
  C    2.761770192  -6.408894369  -3.684717917
  H    3.707673292  -6.707341700  -4.197090673
  C    1.750082299  -4.726262059  -2.289720748
  H    0.639965057  -4.750957237  -2.397780317
  C    2.462039596  -4.918718947  -3.639154987
  H    1.856017279  -4.521856978  -4.485047697
  H    3.404018795  -4.327825076  -3.656324924
  O    2.125286100  -3.484033601  -1.681567593
  O    1.633248159  -5.014665408   2.997997434
  O    0.970451481  -3.900139764   0.632993949
  N    1.638223400  -7.275802751  -4.230209779
  C    0.503388614  -6.786453420  -4.988240524
  N   -0.544752444  -7.660640980  -5.242194738
  C   -0.489455709  -8.973728877  -4.852011031
  C    0.658958042  -9.486482542  -4.162037369
  C    1.674067251  -8.621890555  -3.848586616
  O    0.459854345  -5.632668320  -5.402474615
  N   -1.572250785  -9.738394462  -5.199901898
  H   -1.654738271 -10.700407896  -4.898689042
  H   -2.428283204  -9.284635941  -5.517481033
  H    1.935295344  -4.735491004  -6.153142637
  H    2.558492431  -8.950799192  -3.270329218
  H    0.723186713 -10.541472781  -3.891638028
  H    1.736916488  -2.741763785  -2.241925367
  H   -2.094588785  -2.422853509  -4.777403417
  H   -3.127936040  -3.218255171  -1.952890969
  H    1.768100303  -3.667078804   0.054435342
  H    3.332409839  -6.599922800 -10.052354551
  H   -0.979040959   1.060895678  -5.698025266

