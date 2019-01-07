%mem=64gb
%nproc=28       
%Chk=snap_184.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_184 

2     1 
  O    2.968426923  -4.276651122  -4.667271506
  C    3.127076878  -3.058620918  -5.408793950
  H    4.128276851  -3.184561877  -5.878803497
  H    3.165498516  -2.220682975  -4.683300119
  C    2.052167261  -2.821186165  -6.473408264
  H    2.066337761  -1.761146810  -6.835002235
  O    2.386779424  -3.577823498  -7.675297832
  C    1.425189284  -4.590162573  -7.940788099
  H    1.213524190  -4.527809272  -9.039604096
  C    0.628157513  -3.252519952  -6.057778935
  H    0.580405100  -3.563797379  -4.982251967
  C    0.233413269  -4.395921270  -7.002126894
  H   -0.691842644  -4.140111346  -7.563528680
  H   -0.035596801  -5.305125953  -6.416867183
  O   -0.206907983  -2.100268483  -6.332138114
  N    2.112177206  -5.909033858  -7.693079222
  C    2.236298321  -6.955266380  -8.625076804
  C    2.967704209  -7.995687241  -7.978699643
  N    3.287827226  -7.552110915  -6.678722332
  C    2.776356760  -6.310922546  -6.517535639
  N    1.754621408  -6.994213387  -9.905629902
  C    2.042591267  -8.145142701 -10.593304966
  N    2.762924306  -9.227573481 -10.027673367
  C    3.269961537  -9.221635472  -8.666037734
  N    1.587999111  -8.246775742 -11.881857616
  H    1.093877910  -7.461449250 -12.302184854
  H    1.797479116  -9.030910631 -12.481016276
  O    3.854967551 -10.215901023  -8.293220531
  H    2.873037449  -5.622907505  -5.600783248
  H    2.955980897 -10.077125271 -10.578361380
  P   -1.358659873  -1.833422882  -5.181528570
  O   -2.357495433  -0.758775822  -5.922672792
  C   -1.805174352   0.550861732  -6.183336787
  H   -2.609606500   1.026674735  -6.783620596
  H   -0.901541701   0.473689585  -6.822542574
  C   -1.536091161   1.324554706  -4.884395693
  H   -0.499505422   1.171605278  -4.484018781
  O   -1.577234821   2.734510859  -5.239418524
  C   -2.541507025   3.446645297  -4.460116894
  H   -1.989431278   4.339740384  -4.063248645
  C   -2.593162008   1.103236069  -3.772144787
  H   -3.420209985   0.429939072  -4.098346354
  C   -3.089378483   2.508085791  -3.390222600
  H   -2.668823897   2.767367498  -2.383354003
  H   -4.181719807   2.557459520  -3.228175670
  O   -1.905566797   0.599953583  -2.619339893
  O   -0.645991806  -1.374342078  -3.930788797
  O   -2.113793265  -3.251627697  -5.258853727
  N   -3.560341163   3.966872627  -5.426644313
  C   -4.683720217   3.376409011  -6.001200193
  C   -5.203700894   4.319945125  -6.951454599
  N   -4.388039193   5.460663626  -6.983384646
  C   -3.418507495   5.245085076  -6.099596788
  N   -5.317600675   2.151548379  -5.811749362
  C   -6.508567924   1.944448416  -6.530728794
  N   -7.020812890   2.784809487  -7.416993080
  C   -6.404515085   4.027009800  -7.670679634
  H   -7.038909228   0.985160433  -6.338000788
  N   -6.980998609   4.855686546  -8.558001218
  H   -7.846447734   4.605325402  -9.034629969
  H   -6.571089568   5.765968554  -8.768024774
  H   -2.596912182   5.919970174  -5.867429564
  P   -2.150154697  -1.003744965  -2.288426190
  O   -0.842148002  -1.418906679  -1.416295080
  C   -0.355707252  -2.764267581  -1.607350297
  H    0.098333035  -2.859302412  -2.614407883
  H    0.460617579  -2.815575902  -0.852797723
  C   -1.474387835  -3.768694106  -1.306310583
  H   -1.296176635  -4.299140660  -0.340089466
  O   -2.687491724  -2.973008190  -1.128742194
  C   -3.865497335  -3.836446613  -1.383054373
  H   -4.249006932  -4.095502711  -0.364228466
  C   -1.857659021  -4.775937305  -2.420531468
  H   -1.661980867  -4.422375494  -3.462232475
  C   -3.348162583  -5.045179313  -2.167091029
  H   -3.901714190  -5.243058734  -3.115277637
  H   -3.466866460  -5.990436169  -1.588884651
  O   -1.182254567  -6.035195696  -2.169877506
  O   -2.850608778  -1.711880152  -3.447201663
  O   -3.198125032  -0.713462754  -1.041115666
  N   -4.872604201  -2.961406377  -2.044924095
  C   -4.870974170  -2.799430068  -3.465917694
  N   -5.388087624  -1.582359576  -3.993164005
  C   -5.762599394  -0.476584860  -3.182803526
  C   -5.938330600  -0.777606228  -1.762396928
  C   -5.442890315  -1.930547697  -1.240469141
  O   -4.563538218  -3.699832613  -4.238807385
  H   -5.262316278  -1.447038941  -5.005930445
  O   -5.883208676   0.611313356  -3.743117473
  C   -6.600583542   0.255973594  -0.920676939
  H   -5.893983311   0.699063948  -0.199481289
  H   -7.000143851   1.088582607  -1.523232534
  H   -7.449105912  -0.153623713  -0.350597512
  H   -5.474350723  -2.136746922  -0.161592518
  P    0.349581655  -6.102728857  -2.738788281
  O    1.134987680  -5.260970854  -1.554518462
  C    2.570622374  -5.421322390  -1.487994380
  H    3.027809061  -5.744667433  -2.452119013
  H    2.920147495  -4.386940665  -1.268264741
  C    2.887386775  -6.365139491  -0.327738637
  H    2.213085565  -6.169570261   0.541273711
  O    2.563178835  -7.747575294  -0.677591136
  C    3.751453692  -8.582910199  -0.709388946
  H    3.563069443  -9.333222281   0.101256761
  C    4.394171008  -6.341262764   0.052273959
  H    4.946264214  -5.463686024  -0.362716058
  C    4.957539489  -7.682032865  -0.447634524
  H    5.570097439  -7.523650874  -1.363102946
  H    5.664817019  -8.121414258   0.284580541
  O    4.553887549  -6.149817571   1.443228683
  O    0.590951002  -5.433911815  -4.048371119
  O    0.701401486  -7.663178360  -2.614366011
  N    3.763896777  -9.276434997  -2.026877452
  C    3.902611251  -8.553413490  -3.298029428
  N    3.972783475  -9.292012624  -4.464150096
  C    3.972486275 -10.663042720  -4.442157723
  C    3.823547128 -11.383019413  -3.194948516
  C    3.719301879 -10.678891454  -2.033087526
  O    3.926714739  -7.335213211  -3.282645915
  N    4.111545348 -11.303548421  -5.632601179
  H    4.125731492 -12.310747808  -5.703384026
  H    4.214652369 -10.778114429  -6.507493761
  H    2.041515088  -4.361271106  -4.271850995
  H    3.604221520 -11.186057854  -1.062010561
  H    3.794996195 -12.473484904  -3.193446913
  H    4.198305111  -6.894091348   1.977431041
  H   -3.088241541  -3.299714770  -4.907493611
  H   -2.845047561  -0.902519436  -0.125994542
  H    1.559244127  -7.913712347  -2.067466808
  H    3.770357729  -8.137772055  -5.953099627
  H   -5.059261419   1.492874282  -5.038277638
