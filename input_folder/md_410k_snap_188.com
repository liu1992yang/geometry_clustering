%mem=64gb
%nproc=28       
%Chk=snap_188.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_188 

2     1 
  O    0.451887251  -1.642225459  -6.027063587
  C    0.636699074  -2.690034839  -6.973519393
  H    0.643239440  -3.670973531  -6.452825061
  H    1.650478140  -2.499754599  -7.384183117
  C   -0.436471076  -2.590505464  -8.065459097
  H   -0.537971488  -1.554382377  -8.459048797
  O    0.050527024  -3.365015613  -9.213744186
  C   -0.834142951  -4.435009001  -9.517422141
  H   -0.800304753  -4.574586615 -10.624632837
  C   -1.813013572  -3.200793794  -7.690204992
  H   -1.779696722  -3.769570478  -6.731757723
  C   -2.182204379  -4.094266134  -8.883157479
  H   -2.829214598  -3.528809534  -9.596360673
  H   -2.801173012  -4.969258760  -8.603912362
  O   -2.785947036  -2.127643927  -7.652504671
  N   -0.211295418  -5.659442897  -8.867175826
  C   -0.833931224  -6.884719643  -8.555982668
  C    0.103807937  -7.626042139  -7.776955791
  N    1.271813604  -6.863326215  -7.643778008
  C    1.074332516  -5.683264931  -8.276268170
  N   -2.090810819  -7.309387896  -8.883131689
  C   -2.434483970  -8.542066841  -8.367243145
  N   -1.581158110  -9.306842375  -7.540022240
  C   -0.240788597  -8.895427939  -7.191581974
  N   -3.682532127  -9.024710483  -8.648859712
  H   -4.285505670  -8.522661086  -9.296791041
  H   -4.004206158  -9.925290278  -8.313771358
  O    0.385580868  -9.596450645  -6.426610921
  H    1.791008250  -4.855489735  -8.350633423
  H   -1.949085335 -10.139257048  -7.008445459
  P   -3.267331651  -1.703897340  -6.135941338
  O   -3.518499004  -0.085147318  -6.231661959
  C   -2.357811166   0.761073539  -6.352399060
  H   -2.823942099   1.770831892  -6.391145274
  H   -1.845511982   0.569261329  -7.316451643
  C   -1.404795744   0.611820100  -5.158403369
  H   -0.726064375  -0.287518064  -5.259211099
  O   -0.472601595   1.730780478  -5.214251111
  C   -0.313582911   2.327701114  -3.912420888
  H    0.722622546   2.058660915  -3.568738844
  C   -2.087741167   0.648005317  -3.771650246
  H   -3.200536643   0.713025029  -3.827398711
  C   -1.447185183   1.824279209  -3.017427169
  H   -1.065167698   1.474342407  -2.026639045
  H   -2.202935715   2.598394636  -2.780383686
  O   -1.668506476  -0.524325229  -3.039671182
  O   -2.098093200  -2.073484969  -5.219316103
  O   -4.720447111  -2.377841563  -6.076548455
  N   -0.358919033   3.799646804  -4.172992637
  C   -0.062456933   4.426658700  -5.379071997
  C   -0.096836990   5.837774236  -5.132112281
  N   -0.407225880   6.068509242  -3.784396595
  C   -0.555186247   4.870834092  -3.220392241
  N    0.238031751   3.915077698  -6.640583703
  C    0.519555196   4.863694529  -7.641149049
  N    0.500323067   6.174033419  -7.460258929
  C    0.195945786   6.743164556  -6.203393164
  H    0.772378937   4.472016594  -8.652248404
  N    0.206412680   8.079826361  -6.093280694
  H    0.428993006   8.679595921  -6.889533548
  H   -0.001319144   8.538150774  -5.204612616
  H   -0.781041385   4.671371398  -2.176139147
  P   -2.663944492  -1.841413652  -3.078937996
  O   -1.542172955  -3.015361491  -3.103740955
  C   -2.030275022  -4.374316274  -2.933553572
  H   -3.034657376  -4.528915515  -3.373472715
  H   -1.292199081  -4.961482108  -3.526057003
  C   -2.015550990  -4.734079734  -1.442210439
  H   -2.364414001  -3.889521172  -0.787983890
  O   -3.020121034  -5.773162814  -1.280547831
  C   -2.432782681  -7.002699238  -0.831671245
  H   -3.150169868  -7.384255940  -0.062525269
  C   -0.676957179  -5.311018429  -0.914766381
  H    0.115231355  -5.391786099  -1.697121100
  C   -1.036955678  -6.677264282  -0.299500079
  H   -0.271902820  -7.444988738  -0.542363167
  H   -1.055414666  -6.628022881   0.811386424
  O   -0.288542548  -4.360746395   0.096442425
  O   -3.912458815  -1.682379675  -3.920171937
  O   -3.113666269  -1.947741243  -1.508792575
  N   -2.440283692  -7.959929113  -1.994972816
  C   -1.427229756  -7.934349291  -2.995723591
  N   -1.560722238  -8.871313150  -4.050566503
  C   -2.697669485  -9.720768015  -4.254398760
  C   -3.700708761  -9.667894459  -3.184990344
  C   -3.554169959  -8.822370586  -2.133410736
  O   -0.484786172  -7.159928845  -2.981369711
  H   -0.749327512  -8.951804959  -4.687246418
  O   -2.743491876 -10.361738227  -5.291399539
  C   -4.872206777 -10.577191421  -3.318250730
  H   -5.799807128 -10.021608801  -3.525570335
  H   -4.740782530 -11.299189862  -4.144061995
  H   -5.041523949 -11.178649099  -2.411325540
  H   -4.303981423  -8.769024972  -1.329765254
  P    1.246113844  -4.487482022   0.694648911
  O    1.648557042  -2.900284015   0.598496699
  C    2.926589766  -2.556630681   0.020829395
  H    3.198895058  -1.656753447   0.619471569
  H    3.693603509  -3.346374115   0.178375905
  C    2.812776493  -2.206295581  -1.465398341
  H    3.397128400  -1.277370228  -1.691228503
  O    3.507544065  -3.273667272  -2.185251971
  C    2.856333462  -3.547062488  -3.438916280
  H    3.559373024  -3.207545028  -4.240340759
  C    1.403259311  -2.096635897  -2.083314929
  H    0.585436460  -2.469539746  -1.427715755
  C    1.500868545  -2.834868993  -3.426110398
  H    0.646140484  -3.516132853  -3.588322518
  H    1.450975644  -2.109311272  -4.276350196
  O    1.181889531  -0.691452852  -2.304975893
  O    2.113473861  -5.496279806   0.075601069
  O    0.938903762  -4.641347968   2.282488344
  N    2.716917988  -5.040588196  -3.557057694
  C    2.355563789  -5.575059373  -4.856630894
  N    2.231924176  -6.941533988  -5.013605785
  C    2.419943319  -7.778601242  -3.937882908
  C    2.830104942  -7.261123091  -2.663138621
  C    2.983830835  -5.908200375  -2.500121125
  O    2.162708173  -4.796225857  -5.790350860
  N    2.175691917  -9.106944617  -4.155248797
  H    2.390608663  -9.805068409  -3.460618317
  H    1.907281568  -9.439647153  -5.080409090
  H   -0.343738260  -1.843272200  -5.432753946
  H    3.322689360  -5.454749228  -1.544202433
  H    3.004769580  -7.926792192  -1.817696897
  H    0.282818165  -0.586216560  -2.698313279
  H   -5.231395539  -2.263168834  -5.183186277
  H   -3.988702427  -1.573277281  -1.227314892
  H    0.501036215  -3.873925388   2.751681487
  H    2.013412354  -7.089409957  -6.931621299
  H    0.258960571   2.891806153  -6.782847419

