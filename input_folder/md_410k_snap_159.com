%mem=64gb
%nproc=28       
%Chk=snap_159.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_159 

2     1 
  O    1.903031681  -2.703807711  -4.661407560
  C    1.500510232  -3.759121417  -5.526767019
  H    0.906848794  -4.500352503  -4.956022348
  H    2.471216275  -4.237395517  -5.803373116
  C    0.785839246  -3.211417496  -6.766525520
  H    1.069032911  -2.146039886  -6.951979150
  O    1.316241325  -3.918507743  -7.940721029
  C    0.276372323  -4.507169654  -8.708831149
  H    0.551959456  -4.378272938  -9.782644523
  C   -0.751134932  -3.416497998  -6.809613238
  H   -1.131077373  -4.123211521  -6.037322853
  C   -1.047303765  -3.907572398  -8.234890783
  H   -1.356408072  -3.050294909  -8.875217531
  H   -1.910269314  -4.603413248  -8.279762441
  O   -1.405126808  -2.136430333  -6.700441412
  N    0.332165600  -5.987286303  -8.387174075
  C   -0.653921386  -6.960474428  -8.657420556
  C   -0.289495587  -8.117906635  -7.909539298
  N    0.915639700  -7.850592535  -7.247781575
  C    1.271210908  -6.570385964  -7.512969312
  N   -1.748790346  -6.860315976  -9.469495194
  C   -2.550006118  -7.979246181  -9.486029964
  N   -2.283145472  -9.152279264  -8.738518116
  C   -1.160080623  -9.264665296  -7.824242994
  N   -3.669522976  -7.938041276 -10.277242018
  H   -3.836587713  -7.126420211 -10.867616266
  H   -4.295272130  -8.722316141 -10.385402357
  O   -1.076981603 -10.244491135  -7.119698373
  H    2.164256131  -6.047899440  -7.120642894
  H   -2.962197299  -9.925986389  -8.716066778
  P   -1.848685199  -1.680087898  -5.178546410
  O   -2.846411246  -0.445245199  -5.608282662
  C   -2.194044713   0.680165323  -6.230857910
  H   -3.065377180   1.325528284  -6.478305088
  H   -1.699042048   0.368620372  -7.173767806
  C   -1.221875461   1.387261343  -5.273310786
  H   -0.195928899   0.928377428  -5.253865525
  O   -0.998264768   2.724529105  -5.829809851
  C   -1.270908650   3.746427419  -4.863563870
  H   -0.369317040   4.413503151  -4.877346281
  C   -1.765953462   1.581077190  -3.835390772
  H   -2.823275440   1.234227813  -3.723358967
  C   -1.596078329   3.080210573  -3.531385337
  H   -0.753632588   3.204190178  -2.803413269
  H   -2.478358135   3.486578113  -3.004815341
  O   -0.867351733   0.917272668  -2.930236338
  O   -0.607500344  -1.282730345  -4.392826793
  O   -2.670619572  -2.993806072  -4.707129179
  N   -2.434135282   4.526629673  -5.414256634
  C   -2.829491486   4.592282543  -6.744755990
  C   -3.925300947   5.513349445  -6.809111594
  N   -4.189301242   6.014457783  -5.527372991
  C   -3.308601983   5.437807445  -4.709900537
  N   -2.343539714   3.961080949  -7.889717600
  C   -2.981920506   4.296407377  -9.098014807
  N   -4.004676694   5.129540318  -9.202632284
  C   -4.536292343   5.792328992  -8.073737215
  H   -2.592315647   3.818369972 -10.024543171
  N   -5.568109707   6.631822297  -8.247154225
  H   -5.972930430   6.804344735  -9.167665384
  H   -5.973209506   7.135488140  -7.455693251
  H   -3.212558653   5.618095373  -3.641691975
  P   -1.397184710  -0.539634313  -2.360608318
  O   -0.040585082  -1.229856159  -1.802839766
  C    0.153661346  -2.653108168  -1.964984008
  H   -0.086109380  -2.995187071  -2.991200868
  H    1.257159569  -2.712709138  -1.806696974
  C   -0.620683970  -3.422187238  -0.894229061
  H   -0.777409554  -2.816879932   0.032042306
  O   -1.965184041  -3.681529855  -1.407229474
  C   -2.301726206  -5.070071432  -1.320256944
  H   -3.321856636  -5.088230699  -0.862267028
  C   -0.010945497  -4.808607999  -0.558584284
  H    0.753365091  -5.141183792  -1.318095518
  C   -1.215609244  -5.763053844  -0.497172208
  H   -0.956831528  -6.782433239  -0.852798937
  H   -1.554078332  -5.907887365   0.553093066
  O    0.613625971  -4.628731434   0.723051824
  O   -2.516645994  -1.205000636  -3.132351455
  O   -2.018947889  -0.041339408  -0.930497929
  N   -2.408630882  -5.589870172  -2.729913900
  C   -1.256265677  -5.810896987  -3.533819317
  N   -1.465623347  -6.429962332  -4.790021437
  C   -2.755278271  -6.735419879  -5.345192331
  C   -3.892823136  -6.383334436  -4.484035937
  C   -3.699397263  -5.851071565  -3.251849558
  O   -0.125257594  -5.489675706  -3.200608203
  H   -0.615860607  -6.770089259  -5.267203339
  O   -2.771035369  -7.231715913  -6.453578367
  C   -5.253902838  -6.654984842  -5.022836825
  H   -5.772129892  -7.440389235  -4.449011333
  H   -5.892434321  -5.758222944  -5.008100905
  H   -5.221499237  -7.001138467  -6.069772045
  H   -4.544003829  -5.595741034  -2.594101087
  P    1.579790315  -5.875327873   1.230365621
  O    3.012351792  -5.084260261   1.287331432
  C    4.198162115  -5.794985604   0.851498816
  H    4.947537076  -5.458444725   1.605714353
  H    4.086744868  -6.898077671   0.908457423
  C    4.606297362  -5.357799609  -0.558425851
  H    5.709614636  -5.177211641  -0.595264601
  O    4.409618852  -6.500764598  -1.444725518
  C    3.703421288  -6.102646063  -2.633702151
  H    4.450953090  -6.107013709  -3.466840906
  C    3.816962555  -4.159204526  -1.153940880
  H    3.171187368  -3.652132250  -0.395390783
  C    3.061354966  -4.741704334  -2.355384301
  H    1.969062831  -4.850961243  -2.162905862
  H    3.102223401  -4.061586383  -3.238602513
  O    4.698312555  -3.113219899  -1.516063628
  O    1.495168746  -7.112164865   0.440076396
  O    1.170441594  -5.948260052   2.796946535
  N    2.693201786  -7.175681191  -2.921802491
  C    2.180563935  -7.238854299  -4.276351392
  N    1.292380270  -8.249225985  -4.604883547
  C    0.873018002  -9.150940205  -3.653510355
  C    1.350215769  -9.064845516  -2.306774668
  C    2.262769357  -8.092573356  -1.969807543
  O    2.545610547  -6.400480018  -5.095321941
  N   -0.043828743 -10.077712711  -4.074218621
  H   -0.322099866 -10.847543682  -3.482755537
  H   -0.303984881 -10.143040573  -5.059021701
  H    1.137203646  -2.110660650  -4.414427944
  H    2.690286770  -8.002116218  -0.949965816
  H    1.006225293  -9.764522116  -1.545146099
  H    5.290582233  -3.352303260  -2.262663194
  H   -2.893190022  -3.018743857  -3.692786897
  H   -1.395711594   0.130675718  -0.177542552
  H    1.436287268  -5.184283216   3.389947629
  H    1.306086453  -8.445852951  -6.478570613
  H   -1.535842401   3.324040848  -7.809691860

