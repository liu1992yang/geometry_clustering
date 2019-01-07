%mem=64gb
%nproc=28       
%Chk=snap_13.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_13 

2     1 
  O   -0.436981250  -6.418304527  -7.449240777
  C    0.613850075  -5.577867024  -6.962624312
  H    1.539230892  -6.037660389  -7.378322500
  H    0.652199999  -5.642510063  -5.854514702
  C    0.486025856  -4.121744815  -7.426496647
  H    1.072584646  -3.447340588  -6.746896881
  O    1.209942227  -3.971901770  -8.685775846
  C    0.329681293  -3.755502248  -9.773641527
  H    0.822720009  -2.974466865 -10.411067222
  C   -0.960089618  -3.631184095  -7.690167095
  H   -1.755025193  -4.311888683  -7.296461027
  C   -1.050930110  -3.411279524  -9.210364945
  H   -1.299298281  -2.337676016  -9.404536621
  H   -1.883928400  -3.976158113  -9.666554261
  O   -1.174475473  -2.297518041  -7.168463206
  N    0.309849192  -5.031352205 -10.581680870
  C    0.921308843  -5.207936723 -11.844766739
  C    0.805710667  -6.591528919 -12.162730382
  N    0.158616162  -7.229820593 -11.089470215
  C   -0.121908689  -6.295004204 -10.146597932
  N    1.500259987  -4.253877933 -12.628087229
  C    2.029054324  -4.726937597 -13.811745380
  N    1.960282932  -6.080983880 -14.203363617
  C    1.329195741  -7.127589036 -13.399523976
  N    2.642197530  -3.818476087 -14.631049894
  H    2.703443542  -2.842879801 -14.345678641
  H    3.078642874  -4.070421037 -15.506884810
  O    1.319042931  -8.246778915 -13.835043331
  H   -0.590102284  -6.464655398  -9.128619760
  H    2.368036966  -6.393144596 -15.098791588
  P   -1.220745134  -2.115330676  -5.542171406
  O   -1.215533076  -0.459682071  -5.562756316
  C   -0.341234290   0.196060934  -4.617525494
  H    0.343598843   0.813356402  -5.243515141
  H    0.260882293  -0.522671589  -4.016622970
  C   -1.194073664   1.089557378  -3.715842945
  H   -0.703282349   1.229222421  -2.715843836
  O   -1.165909612   2.438531417  -4.293862015
  C   -2.471094780   2.963458634  -4.455222944
  H   -2.420888697   4.003108697  -4.055638828
  C   -2.689710146   0.704845153  -3.559075710
  H   -3.032331841  -0.094421378  -4.256133335
  C   -3.462188054   2.019552340  -3.772803683
  H   -3.784097898   2.410804166  -2.775463662
  H   -4.402510631   1.864261517  -4.332807483
  O   -2.953861272   0.357480808  -2.191775964
  O   -0.161273949  -2.790568950  -4.761276352
  O   -2.728969431  -2.435717419  -5.120207711
  N   -2.730243366   3.034339358  -5.940102544
  C   -3.295849294   4.087748119  -6.650868001
  C   -3.349779299   3.677814996  -8.024960935
  N   -2.809770486   2.388935758  -8.150333506
  C   -2.446831018   2.015384443  -6.922401982
  N   -3.782137753   5.336407418  -6.256678796
  C   -4.314802713   6.150546122  -7.276853253
  N   -4.386421934   5.807141243  -8.550650753
  C   -3.909478396   4.558118770  -9.003344227
  H   -4.705055508   7.148600426  -6.972139581
  N   -4.020702974   4.280287845 -10.313002083
  H   -4.434335629   4.947853169 -10.964568209
  H   -3.692799816   3.393265398 -10.691703388
  H   -1.981621306   1.062012121  -6.638761771
  P   -2.636328812  -1.198401627  -1.728796088
  O   -1.009068485  -1.053975646  -1.893315667
  C   -0.137676406  -2.147036850  -1.516420676
  H   -0.059867977  -2.831126238  -2.395101250
  H    0.840061154  -1.642903598  -1.363150595
  C   -0.652028716  -2.819818696  -0.246744663
  H   -0.635208804  -2.157661623   0.652493864
  O   -2.079565010  -3.054769770  -0.511034096
  C   -2.351396755  -4.493563624  -0.638664868
  H   -3.365384770  -4.594885573  -0.181826490
  C   -0.033489492  -4.203306224   0.068236129
  H    0.749863646  -4.503543781  -0.675158851
  C   -1.222394716  -5.169941108   0.108102854
  H   -0.963972684  -6.197637294  -0.277458860
  H   -1.487796547  -5.411128597   1.169034167
  O    0.497923280  -4.087675271   1.405939877
  O   -3.300964688  -2.220923684  -2.602158674
  O   -3.133040334  -0.982601084  -0.206823138
  N   -2.435108466  -4.836303724  -2.092279931
  C   -1.279829211  -5.155167875  -2.860648168
  N   -1.499420662  -5.513469409  -4.213417779
  C   -2.763399682  -5.466614292  -4.881084565
  C   -3.883166280  -5.055321201  -4.041211781
  C   -3.698220546  -4.740401682  -2.730650598
  O   -0.138059691  -5.136747261  -2.431562944
  H   -0.652716380  -5.754481582  -4.756167886
  O   -2.742552055  -5.735332014  -6.077452589
  C   -5.213499852  -4.937236207  -4.700335287
  H   -5.459771635  -3.880606301  -4.898185217
  H   -5.237746782  -5.457004308  -5.673185737
  H   -6.023882105  -5.367628649  -4.094311623
  H   -4.527929703  -4.382043727  -2.102043897
  P    2.017550462  -4.655023780   1.652773634
  O    2.174052753  -5.662960003   0.374550482
  C    3.515091881  -6.111398280   0.018472057
  H    4.074745721  -5.237365495  -0.369325328
  H    4.015125281  -6.550868232   0.907474437
  C    3.327943145  -7.187381638  -1.061741367
  H    4.147192606  -7.150715554  -1.816757856
  O    3.498459204  -8.479739167  -0.411971876
  C    2.244692826  -9.206595418  -0.417846627
  H    2.540232037 -10.280068286  -0.419708790
  C    1.930840701  -7.228895104  -1.742922235
  H    1.184405671  -6.572155126  -1.239765245
  C    1.521845499  -8.711513779  -1.669551903
  H    0.420125938  -8.847995039  -1.637210426
  H    1.871731631  -9.255887071  -2.569712467
  O    2.071967604  -6.810718436  -3.104899810
  O    2.315533800  -5.082747498   3.019262661
  O    2.923426857  -3.386685749   1.102278230
  N    1.579260168  -8.866482962   0.887894603
  C    0.367009517  -8.041028662   0.992980423
  N    0.138675864  -7.349323011   2.169938159
  C    0.929992760  -7.546316933   3.262064297
  C    2.042846478  -8.467779367   3.223685490
  C    2.362623788  -9.056795000   2.035201662
  O   -0.382657027  -7.968387658   0.028807387
  N    0.593641945  -6.870561045   4.415476463
  H    1.304355688  -6.728285516   5.121714741
  H   -0.073738851  -6.109340750   4.343398699
  H   -1.261602113  -6.340628008  -6.864185033
  H    3.251089169  -9.697367464   1.930941075
  H    2.619293689  -8.662844225   4.125711648
  H    1.421872860  -6.082821708  -3.259839851
  H   -2.927903242  -2.535856184  -4.046571814
  H   -3.246413369  -1.810203440   0.373836806
  H    3.444072149  -2.860477986   1.758853439
  H   -0.048624845  -8.225447696 -11.062477928
  H   -3.729418287   5.651491497  -5.285491052
