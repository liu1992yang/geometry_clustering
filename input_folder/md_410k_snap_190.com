%mem=64gb
%nproc=28       
%Chk=snap_190.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_190 

2     1 
  O    0.276425619  -1.614483327  -5.985468375
  C    0.492969583  -2.676557217  -6.908597919
  H    0.618637186  -3.634068132  -6.360265333
  H    1.455031156  -2.414133598  -7.395751024
  C   -0.656118730  -2.715241873  -7.922871103
  H   -0.861595998  -1.712644750  -8.362124899
  O   -0.198100074  -3.516977055  -9.062761954
  C   -1.056693236  -4.627652766  -9.283378205
  H   -1.115445040  -4.776838678 -10.388971472
  C   -1.956490390  -3.389152583  -7.409864121
  H   -1.818443478  -3.914676952  -6.436421559
  C   -2.363204604  -4.350269091  -8.536792547
  H   -3.108419962  -3.854205977  -9.204968190
  H   -2.889556281  -5.255663028  -8.177939467
  O   -2.978588458  -2.368344926  -7.337852034
  N   -0.329399144  -5.819499246  -8.688022253
  C   -0.843432646  -7.115303012  -8.478343062
  C    0.135455881  -7.817376045  -7.713503760
  N    1.221973817  -6.961508968  -7.492442771
  C    0.936062287  -5.766512477  -8.059086457
  N   -2.044728208  -7.628593182  -8.874118495
  C   -2.261547762  -8.938625760  -8.495090912
  N   -1.337500640  -9.698040725  -7.736861846
  C   -0.091238304  -9.159651674  -7.244982653
  N   -3.427325917  -9.529229744  -8.894973660
  H   -4.086226783  -9.009170332  -9.469712414
  H   -3.719509585 -10.438904885  -8.559552556
  O    0.568363162  -9.827065497  -6.477203223
  H    1.588498031  -4.882071743  -8.060734279
  H   -1.629970434 -10.604658934  -7.318640753
  P   -3.356666357  -1.857764819  -5.817930378
  O   -3.588946559  -0.240760542  -5.994915759
  C   -2.422099270   0.589174415  -6.162602036
  H   -2.882270661   1.589410586  -6.325679687
  H   -1.865792304   0.301990246  -7.077079435
  C   -1.522729427   0.565666983  -4.917718240
  H   -0.830847018  -0.326998180  -4.907542634
  O   -0.599716606   1.688055462  -5.022211879
  C   -0.537741398   2.414603114  -3.776221754
  H    0.461428471   2.171047724  -3.321237301
  C   -2.279233212   0.719371131  -3.577041125
  H   -3.391389977   0.712157156  -3.688587581
  C   -1.745962277   2.008287195  -2.932911067
  H   -1.462668168   1.808928849  -1.870465866
  H   -2.538907711   2.780659545  -2.889236327
  O   -1.833252653  -0.332668039  -2.693240845
  O   -2.163979560  -2.193050028  -4.923019927
  O   -4.820167472  -2.496692263  -5.666916749
  N   -0.534117001   3.847308325  -4.196179103
  C   -0.027881207   4.325938268  -5.400912547
  C   -0.087465757   5.755283259  -5.336833148
  N   -0.625029844   6.144699321  -4.101638561
  C   -0.885108556   5.023138540  -3.430893878
  N    0.477505358   3.663745138  -6.518286628
  C    0.939152694   4.482827327  -7.563837511
  N    0.908193159   5.805756157  -7.549696774
  C    0.400557363   6.522945633  -6.443700155
  H    1.360057536   3.971112055  -8.458357990
  N    0.415602331   7.863121525  -6.495643376
  H    0.783519385   8.361170077  -7.307593892
  H    0.066950723   8.424693101  -5.717289437
  H   -1.296423969   4.951932224  -2.427064096
  P   -2.681717268  -1.748983087  -2.739757566
  O   -1.390491347  -2.731097981  -2.753986519
  C   -1.638802141  -4.160101965  -2.718747308
  H   -2.528122852  -4.451713493  -3.310524320
  H   -0.729002001  -4.551285659  -3.234653443
  C   -1.762334654  -4.621645519  -1.261596120
  H   -2.038735672  -3.788663838  -0.558024424
  O   -2.904129141  -5.523107490  -1.225145654
  C   -2.494808286  -6.864863376  -0.946447190
  H   -3.284337022  -7.257686729  -0.258639384
  C   -0.546749743  -5.415054991  -0.718247209
  H    0.304196197  -5.469635702  -1.438635465
  C   -1.090354933  -6.806453024  -0.345164439
  H   -0.415920391  -7.614540661  -0.690334339
  H   -1.137953138  -6.912234736   0.761472653
  O   -0.196377122  -4.747125901   0.513883345
  O   -3.925713585  -1.688374963  -3.594222387
  O   -3.122944377  -1.950563300  -1.181016085
  N   -2.562913180  -7.643774961  -2.235698439
  C   -1.498589005  -7.630037516  -3.182701951
  N   -1.646564051  -8.487778631  -4.301041722
  C   -2.826927942  -9.249162954  -4.601671207
  C   -3.915305137  -9.104964345  -3.629476621
  C   -3.757380517  -8.346846590  -2.515256675
  O   -0.502862145  -6.934531753  -3.068378455
  H   -0.815295091  -8.594150806  -4.899866955
  O   -2.819685921  -9.902096906  -5.629508468
  C   -5.178621272  -9.840943486  -3.911831320
  H   -5.156293192 -10.336226211  -4.897428818
  H   -5.360738504 -10.637129229  -3.170814888
  H   -6.057196607  -9.178102526  -3.906656693
  H   -4.559488982  -8.246727141  -1.768330885
  P    1.411969731  -4.543097667   0.791251951
  O    1.555248767  -2.967299541   0.400235139
  C    2.873332749  -2.506124754   0.005707477
  H    3.023676459  -1.621164770   0.664430038
  H    3.680668112  -3.245898052   0.188955899
  C    2.838100053  -2.101730214  -1.470794703
  H    3.420737069  -1.161271670  -1.638587681
  O    3.575035944  -3.132155184  -2.202438167
  C    2.931715760  -3.435331207  -3.450258896
  H    3.634688843  -3.102380638  -4.254063800
  C    1.442882090  -1.996619152  -2.122146381
  H    0.618470504  -2.376857151  -1.476436545
  C    1.564335493  -2.748491709  -3.455045260
  H    0.726716168  -3.456938074  -3.601219390
  H    1.495655111  -2.034105889  -4.311520233
  O    1.215451694  -0.596597428  -2.360383470
  O    2.294143100  -5.508156106   0.108084876
  O    1.451144896  -4.472465367   2.413671664
  N    2.805993981  -4.935187399  -3.546579549
  C    2.410338596  -5.483964293  -4.829493589
  N    2.294449336  -6.854238385  -4.966646208
  C    2.514506948  -7.681045982  -3.889387938
  C    2.960520220  -7.147097505  -2.634735489
  C    3.109472715  -5.790162527  -2.489820919
  O    2.176910278  -4.716645079  -5.762783850
  N    2.250041854  -9.009815524  -4.087873759
  H    2.515077259  -9.706155134  -3.407163781
  H    1.971592771  -9.351067181  -5.008277665
  H   -0.465334762  -1.859532552  -5.336237961
  H    3.468089982  -5.324726519  -1.546288074
  H    3.172696119  -7.801056967  -1.788063087
  H    0.258684008  -0.476388646  -2.568599787
  H   -5.291953018  -2.327977142  -4.762031096
  H   -4.052378982  -1.726146983  -0.905613937
  H    1.435867120  -5.324756963   2.929881874
  H    1.972811603  -7.153973911  -6.769383716
  H    0.492385348   2.630379424  -6.535575056
