%mem=64gb
%nproc=28       
%Chk=snap_31.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_31 

2     1 
  O    2.190614374  -4.381084836  -7.656509431
  C    2.258570214  -2.984145523  -8.006276949
  H    3.165446869  -2.929564650  -8.646953871
  H    2.417832576  -2.407411597  -7.067979122
  C    1.023233503  -2.480908823  -8.751900780
  H    1.052336984  -1.367402212  -8.855232556
  O    1.089102145  -2.954689501 -10.139152167
  C   -0.107347948  -3.619936185 -10.512372949
  H   -0.330507597  -3.311682409 -11.563799318
  C   -0.350698769  -2.948044321  -8.199958979
  H   -0.263067536  -3.826507154  -7.512219227
  C   -1.166203282  -3.314462946  -9.450370622
  H   -1.807385485  -2.453626159  -9.747080773
  H   -1.871618433  -4.146192588  -9.266545791
  O   -0.985084438  -1.825980033  -7.572624947
  N    0.218207675  -5.097991070 -10.537202087
  C   -0.328838020  -6.055204419 -11.418363927
  C    0.263005715  -7.304136213 -11.079698877
  N    1.151104057  -7.081900383 -10.007080099
  C    1.114337381  -5.764411827  -9.690960634
  N   -1.251036644  -5.849613868 -12.406690936
  C   -1.590640485  -6.980629175 -13.113167712
  N   -1.049690295  -8.256843003 -12.845143348
  C   -0.069151646  -8.515450282 -11.790700667
  N   -2.519519345  -6.843102626 -14.112331773
  H   -2.894104353  -5.922022440 -14.326633999
  H   -2.792264262  -7.603131025 -14.717300254
  O    0.320191171  -9.644286254 -11.651838299
  H    1.710634800  -5.209931273  -8.840604007
  H   -1.332874918  -9.077068576 -13.401501033
  P   -0.860866203  -1.828660416  -5.906013844
  O   -1.285277293  -0.236836698  -5.758280284
  C   -0.358189024   0.584123643  -5.006485707
  H    0.036749861   1.326804690  -5.735871296
  H    0.493876675   0.001624153  -4.586562662
  C   -1.167161885   1.277034595  -3.907757349
  H   -0.583328280   1.374231081  -2.957920146
  O   -1.360540945   2.671640361  -4.322925377
  C   -2.725943259   3.046419418  -4.240388590
  H   -2.728709634   4.049876936  -3.756812986
  C   -2.588300714   0.693910492  -3.664174542
  H   -2.942257082   0.020569573  -4.483913797
  C   -3.485645030   1.935087550  -3.513480069
  H   -3.611995772   2.159660810  -2.427053049
  H   -4.509462246   1.758196037  -3.890414013
  O   -2.663907416   0.010086818  -2.414376747
  O    0.489110386  -2.241802133  -5.469310738
  O   -2.181543845  -2.732357893  -5.636521799
  N   -3.219246422   3.186774185  -5.657915954
  C   -3.973198975   4.225119525  -6.193867750
  C   -4.227495505   3.885086297  -7.563833204
  N   -3.618437113   2.657653570  -7.863589321
  C   -3.025037578   2.250420318  -6.741535619
  N   -4.476705545   5.406964346  -5.645502725
  C   -5.229872846   6.226311976  -6.511691978
  N   -5.492230031   5.943498967  -7.775057703
  C   -5.008412463   4.762681706  -8.379709116
  H   -5.631004875   7.173456811  -6.084038888
  N   -5.316533755   4.536220380  -9.666729843
  H   -5.885402379   5.193128586 -10.201129730
  H   -4.980149989   3.702108784 -10.148468338
  H   -2.441994951   1.330000906  -6.607759785
  P   -1.821721947  -1.424169404  -2.293434986
  O   -0.725359303  -0.727976448  -1.310025073
  C    0.427172238  -1.476486598  -0.853574656
  H    1.025658226  -1.835173338  -1.722105413
  H    0.999549018  -0.711647854  -0.288336980
  C   -0.112486029  -2.597386663   0.029945506
  H   -0.590362413  -2.233762934   0.971035847
  O   -1.222148851  -3.120755295  -0.775584722
  C   -1.032207069  -4.540718937  -1.080024497
  H   -2.023353006  -4.992184240  -0.818155941
  C    0.853472890  -3.768721742   0.308816346
  H    1.863521421  -3.623315713  -0.150645196
  C    0.131156667  -5.013744398  -0.217838994
  H    0.852899377  -5.656302338  -0.786595050
  H   -0.262071661  -5.673463287   0.600339528
  O    1.033566412  -3.741734016   1.738590262
  O   -1.351996903  -1.948269122  -3.615158927
  O   -3.078501013  -2.267540122  -1.704081261
  N   -0.794810613  -4.662680115  -2.540411048
  C    0.488739821  -4.343019742  -3.096704026
  N    0.679122053  -4.683231276  -4.451987648
  C   -0.307053809  -5.287609906  -5.280170555
  C   -1.589228234  -5.578304447  -4.640876834
  C   -1.782607707  -5.306248136  -3.324246543
  O    1.370679700  -3.819430316  -2.437940265
  H    1.548472329  -4.327923798  -4.889975825
  O    0.003465214  -5.459182034  -6.459740441
  C   -2.646502144  -6.166936043  -5.508274255
  H   -3.565883388  -6.413919229  -4.960498774
  H   -2.916936849  -5.472856610  -6.321464958
  H   -2.295608131  -7.102737789  -5.976580624
  H   -2.715426886  -5.568087290  -2.802706921
  P    1.500934828  -5.133819648   2.485573808
  O    2.873864310  -5.494579709   1.677326014
  C    3.520703142  -6.737366098   2.085120531
  H    4.450678847  -6.397846629   2.595249205
  H    2.906843445  -7.345566113   2.780975292
  C    3.848426043  -7.545593237   0.827916097
  H    4.768657556  -8.156818230   1.007460294
  O    2.820581346  -8.555670844   0.625197934
  C    2.138057825  -8.383056465  -0.629056305
  H    2.265799006  -9.353968578  -1.167980449
  C    3.950886224  -6.722200677  -0.481306391
  H    3.988857895  -5.620833876  -0.299513917
  C    2.752552543  -7.167684707  -1.331873100
  H    2.023095574  -6.334876741  -1.456461773
  H    3.049582196  -7.404524784  -2.373733012
  O    5.189005438  -6.963337509  -1.125841301
  O    0.484226136  -6.198279087   2.554876677
  O    2.041813001  -4.530664025   3.896218246
  N    0.673284560  -8.217837469  -0.317848055
  C   -0.249574233  -7.844992607  -1.408895489
  N   -1.592291244  -7.710055300  -1.102795716
  C   -2.045711306  -8.003893745   0.152490244
  C   -1.162562426  -8.395222586   1.209153884
  C    0.184916546  -8.478235685   0.956648716
  O    0.246811674  -7.607774905  -2.495930222
  N   -3.396534137  -7.808663669   0.379600965
  H   -3.819831796  -8.182336141   1.217652183
  H   -4.018450097  -7.703235996  -0.411708437
  H    1.451966867  -4.527664113  -6.975385928
  H    0.915007630  -8.747813956   1.738407700
  H   -1.537599177  -8.577700640   2.215142890
  H    5.297956056  -7.901349406  -1.397090296
  H   -2.313956373  -3.033811549  -4.660921104
  H   -3.135143294  -2.414823054  -0.709721722
  H    1.402357982  -4.470412418   4.661205948
  H    1.705061360  -7.812212995  -9.570899681
  H   -4.280965235   5.678435371  -4.679886725
