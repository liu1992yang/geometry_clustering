%mem=64gb
%nproc=28       
%Chk=snap_134.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_134 

2     1 
  O    1.631856860  -4.393211856  -5.554137677
  C    1.977126238  -3.109122853  -6.092852473
  H    2.998731994  -3.111332459  -6.514652051
  H    1.938586756  -2.414009109  -5.228059824
  C    0.926779610  -2.730344690  -7.138951216
  H    0.901496626  -1.634293320  -7.342717268
  O    1.310211246  -3.301920355  -8.435348877
  C    0.339699043  -4.237266631  -8.880956758
  H    0.300854682  -4.151658326  -9.993640022
  C   -0.474422019  -3.307585237  -6.802727261
  H   -0.435698376  -4.028841044  -5.942262880
  C   -0.948043497  -3.969938291  -8.104219056
  H   -1.618528673  -3.273390278  -8.663098230
  H   -1.582550002  -4.862089676  -7.938475865
  O   -1.367365293  -2.210681252  -6.538333374
  N    0.936668028  -5.599681300  -8.549221250
  C    0.280324984  -6.833355639  -8.394879498
  C    1.258173228  -7.761005622  -7.922680167
  N    2.479919015  -7.084772439  -7.796474200
  C    2.282092709  -5.796151201  -8.168371914
  N   -1.027962243  -7.135813711  -8.658215043
  C   -1.370603108  -8.451668133  -8.459263170
  N   -0.462881577  -9.429784510  -7.983871881
  C    0.928012917  -9.142856929  -7.671846523
  N   -2.670878763  -8.809453785  -8.701538497
  H   -3.312658721  -8.122507286  -9.097572272
  H   -2.988147855  -9.769213254  -8.685213847
  O    1.613298271 -10.049735636  -7.262226454
  H    3.039683714  -5.001649579  -8.204704428
  H   -0.767120854 -10.404830117  -7.837582167
  P   -1.541252788  -1.818596472  -4.936796923
  O   -0.851569602  -0.300886292  -5.024788692
  C    0.068300461   0.103338701  -3.992329141
  H    0.913960996   0.547673121  -4.570801658
  H    0.439198714  -0.762583811  -3.401724102
  C   -0.551556579   1.168360014  -3.091883158
  H    0.044327391   1.262429651  -2.144750187
  O   -0.382403974   2.449103342  -3.782532493
  C   -1.480938355   3.325269374  -3.499607763
  H   -1.033776500   4.239575904  -3.032579431
  C   -2.064597012   1.082920617  -2.761202260
  H   -2.666688538   0.562079472  -3.542268662
  C   -2.484121537   2.555600445  -2.630613286
  H   -2.427073209   2.866333261  -1.559497376
  H   -3.541341127   2.727341536  -2.903421136
  O   -2.233227178   0.479947101  -1.474886985
  O   -0.853968898  -2.766588071  -4.041645269
  O   -3.074681981  -1.312082972  -5.147915170
  N   -2.071320357   3.735371613  -4.811105785
  C   -2.552629497   2.951475627  -5.856668691
  C   -3.154830308   3.845651857  -6.803254301
  N   -3.022265284   5.167359726  -6.351401600
  C   -2.384534725   5.100842607  -5.185588123
  N   -2.548570794   1.580494973  -6.079311742
  C   -3.179579243   1.127341812  -7.247027049
  N   -3.765769976   1.914617638  -8.140965959
  C   -3.774213085   3.314288814  -7.979048574
  H   -3.195751752   0.021639048  -7.409371503
  N   -4.368484112   4.063981376  -8.924536331
  H   -4.802687863   3.642509051  -9.743963758
  H   -4.397378435   5.080442000  -8.844928019
  H   -2.105474669   5.941641763  -4.554394365
  P   -2.465998209  -1.169552478  -1.502065181
  O   -1.081118657  -1.568351153  -0.756671800
  C   -0.634388517  -2.944541335  -0.926654686
  H   -0.523091661  -3.188150544  -2.012139664
  H    0.374379134  -2.938145525  -0.449522121
  C   -1.637308085  -3.821355993  -0.192296307
  H   -1.638939661  -3.652821055   0.911149378
  O   -2.938850056  -3.305219331  -0.647111164
  C   -3.790186097  -4.390373306  -1.142513302
  H   -4.768356588  -4.193268926  -0.634764654
  C   -1.619433125  -5.325408821  -0.543780470
  H   -1.008860663  -5.569723446  -1.454981935
  C   -3.104009252  -5.688152400  -0.723946077
  H   -3.237893518  -6.522884235  -1.442673319
  H   -3.508651473  -6.075669584   0.240827533
  O   -1.187056745  -6.076516270   0.609186200
  O   -2.660531158  -1.583383868  -2.935250917
  O   -3.789729393  -1.187228418  -0.569314665
  N   -3.964093809  -4.226689605  -2.610136185
  C   -3.048198609  -4.802449595  -3.554644377
  N   -3.373567756  -4.613323225  -4.919092310
  C   -4.509941598  -3.883939563  -5.427553309
  C   -5.310202892  -3.232867832  -4.376572600
  C   -5.026781304  -3.405944531  -3.061243061
  O   -2.081168147  -5.465731321  -3.238298964
  H   -2.777272217  -5.108949358  -5.596678110
  O   -4.655851286  -3.874376040  -6.627470330
  C   -6.462932819  -2.410701767  -4.838670905
  H   -6.414677894  -2.213246112  -5.924529782
  H   -7.421879729  -2.930020277  -4.668904902
  H   -6.524719104  -1.437370734  -4.335335525
  H   -5.621714521  -2.916315991  -2.275138726
  P    0.423828837  -6.363623162   0.737812032
  O    0.775793941  -6.497002995  -0.866673224
  C    2.150642302  -6.706670813  -1.238738273
  H    2.703967653  -7.332832862  -0.508143787
  H    2.057075359  -7.281454860  -2.183820478
  C    2.824954431  -5.342805984  -1.438147363
  H    2.713266558  -4.664783957  -0.559515905
  O    4.257479708  -5.642238363  -1.513497765
  C    4.789861523  -5.232826823  -2.784361042
  H    5.845636065  -4.938798236  -2.585277525
  C    2.468402094  -4.610821015  -2.759663535
  H    1.936657848  -5.264872786  -3.484242744
  C    3.846133039  -4.145951031  -3.279985825
  H    3.878316741  -3.982918489  -4.372661484
  H    4.106967979  -3.162661636  -2.833876514
  O    1.642272971  -3.486837143  -2.466781310
  O    0.855632729  -7.389906090   1.673974646
  O    0.999901641  -4.834075869   1.019919869
  N    4.782020271  -6.491448722  -3.637501776
  C    4.292123951  -6.557494380  -4.997556300
  N    4.031685911  -7.794902114  -5.561490146
  C    4.352191980  -8.952676141  -4.894422206
  C    4.986105415  -8.895152616  -3.600110183
  C    5.157146415  -7.680245920  -2.998526297
  O    4.077822167  -5.533876977  -5.647915126
  N    4.030225867 -10.127553771  -5.501653478
  H    4.230527500 -11.021692894  -5.076538016
  H    3.487664110 -10.139942494  -6.365267678
  H    2.319122601  -5.070252414  -5.792152108
  H    5.575708162  -7.582882285  -1.980852810
  H    5.323429609  -9.809246997  -3.110157442
  H    0.904296510  -3.452435626  -3.148372874
  H   -3.719448767  -1.508624505  -4.359970625
  H   -3.758294726  -1.718893413   0.290344749
  H    1.402967363  -4.648592090   1.908254267
  H    3.317986022  -7.484893938  -7.290542775
  H   -2.001863492   0.925377739  -5.460923080

