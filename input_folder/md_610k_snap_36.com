%mem=64gb
%nproc=28       
%Chk=snap_36.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_36 

2     1 
  O    2.087237557  -4.455859265  -7.435082003
  C    2.414709525  -3.255844737  -8.159999480
  H    3.386140651  -3.514828938  -8.639043368
  H    2.582715849  -2.438455626  -7.426403836
  C    1.373466131  -2.867907682  -9.209887000
  H    1.568763215  -1.832363938  -9.592492711
  O    1.577395209  -3.707355352 -10.394616669
  C    0.374191597  -4.349576422 -10.788120704
  H    0.344475596  -4.310676805 -11.906431345
  C   -0.111110849  -3.059577403  -8.801226483
  H   -0.244244003  -3.716582945  -7.897755107
  C   -0.780259362  -3.685493411 -10.031730793
  H   -1.269419790  -2.894471935 -10.644475731
  H   -1.594216428  -4.383253872  -9.757669699
  O   -0.741281779  -1.778133378  -8.590087943
  N    0.512301322  -5.801202561 -10.387244942
  C    0.067846722  -6.916956811 -11.130754953
  C    0.398011777  -8.071415143 -10.367252935
  N    1.033009112  -7.639153456  -9.185194935
  C    1.099209317  -6.284989662  -9.208750754
  N   -0.559032086  -6.916496386 -12.344990861
  C   -0.848599676  -8.169210598 -12.839273810
  N   -0.546993400  -9.365097991 -12.150601109
  C    0.101407843  -9.402474545 -10.840480066
  N   -1.477672503  -8.237746729 -14.054646679
  H   -1.675173998  -7.381159785 -14.568079751
  H   -1.700547495  -9.110534131 -14.511649774
  O    0.301657372 -10.477536659 -10.340700796
  H    1.570169690  -5.587550970  -8.406286810
  H   -0.774596878 -10.283182028 -12.562020974
  P   -0.443436140  -1.141026473  -7.096408495
  O   -1.861716073  -0.337776447  -6.781188020
  C   -1.737127903   1.084291075  -6.593397461
  H   -2.783295249   1.435082403  -6.715877444
  H   -1.105386359   1.556139515  -7.372609311
  C   -1.193921922   1.350259722  -5.179398835
  H   -0.088702862   1.161434497  -5.081335990
  O   -1.307389280   2.782076034  -4.919315153
  C   -2.231410302   3.051100958  -3.860372578
  H   -1.680087204   3.727648541  -3.155257057
  C   -2.003002867   0.619302205  -4.080915388
  H   -2.712471899  -0.135396668  -4.496556452
  C   -2.707660063   1.726193160  -3.271927046
  H   -2.419251904   1.643824317  -2.196398092
  H   -3.804062138   1.583755217  -3.281927425
  O   -0.994446078   0.016628638  -3.255149911
  O    0.751395041  -0.292915195  -6.982400797
  O   -0.548599681  -2.458649325  -6.185922687
  N   -3.335829702   3.849553101  -4.498371172
  C   -3.204514218   4.639891801  -5.635146469
  C   -4.441754801   5.343294862  -5.800109974
  N   -5.321123574   4.985426125  -4.768787049
  C   -4.666197673   4.112968842  -4.001590245
  N   -2.142192730   4.812622402  -6.521382784
  C   -2.352199897   5.731569298  -7.564308051
  N   -3.475006845   6.406601512  -7.755284335
  C   -4.579975722   6.265329272  -6.887559207
  H   -1.510420705   5.891503166  -8.275424336
  N   -5.675864556   7.001115952  -7.129145836
  H   -5.719248241   7.653543792  -7.913179060
  H   -6.494987849   6.939671798  -6.523061280
  H   -5.044019065   3.644262559  -3.096773385
  P   -1.284838050  -1.484127169  -2.629517166
  O   -0.407256906  -1.291723269  -1.282352445
  C    0.101210784  -2.523677456  -0.694747852
  H    0.752297885  -3.061506599  -1.424049603
  H    0.728842479  -2.149315984   0.143467003
  C   -1.107131977  -3.326531422  -0.219125278
  H   -1.651526408  -2.842598850   0.626192693
  O   -2.056783392  -3.269581790  -1.344370029
  C   -2.432613151  -4.629627324  -1.778508160
  H   -3.545905863  -4.570626242  -1.837531029
  C   -0.832015662  -4.822637545   0.057636693
  H    0.208216956  -5.142454081  -0.231880406
  C   -1.896534001  -5.588738290  -0.731326365
  H   -1.467917558  -6.538568369  -1.163315704
  H   -2.684915533  -5.965074644  -0.041291771
  O   -1.132564624  -5.033035577   1.453119946
  O   -0.919442262  -2.520907106  -3.643062196
  O   -2.881984926  -1.283506104  -2.324295784
  N   -1.872607800  -4.861900252  -3.133828934
  C   -0.454255295  -4.973759520  -3.328308612
  N   -0.041733050  -5.245850675  -4.662799104
  C   -0.897159537  -5.217155639  -5.794628576
  C   -2.305512463  -4.921595143  -5.524007606
  C   -2.733869743  -4.735148797  -4.248392270
  O    0.368213391  -4.903266002  -2.437816299
  H    0.969676978  -5.391463381  -4.803059372
  O   -0.367031854  -5.425318485  -6.886693869
  C   -3.197658902  -4.782988743  -6.708188367
  H   -4.262544956  -4.886767064  -6.460157056
  H   -3.062969682  -3.797190445  -7.184493585
  H   -2.973278730  -5.546742913  -7.470331058
  H   -3.781022620  -4.497262540  -4.020996694
  P    0.094278321  -5.413004053   2.469828902
  O    1.337278078  -5.457893145   1.406373943
  C    2.673516230  -5.743870623   1.905724954
  H    3.004199957  -4.874869988   2.509237287
  H    2.664828974  -6.673254385   2.513750396
  C    3.553672927  -5.928775859   0.659081304
  H    4.560703383  -5.481401764   0.828207769
  O    3.839262736  -7.342880826   0.499999038
  C    3.054326977  -7.897178402  -0.584564004
  H    3.680064722  -8.731663819  -0.977942989
  C    2.918096883  -5.462767017  -0.684659293
  H    1.921225293  -4.968412828  -0.558957899
  C    2.825817826  -6.734589747  -1.547383851
  H    1.844805536  -6.779437009  -2.082937369
  H    3.581746362  -6.736116952  -2.352961094
  O    3.693048880  -4.436008966  -1.278032593
  O   -0.137331531  -6.518039433   3.401797003
  O    0.362370030  -3.948887272   3.167817588
  N    1.826576968  -8.463314720   0.068280093
  C    0.451474783  -8.107470106  -0.305114411
  N   -0.534008594  -8.221457698   0.660139127
  C   -0.281867300  -8.834325293   1.851481194
  C    1.026507335  -9.372256428   2.153738287
  C    2.046867534  -9.131437646   1.282286888
  O    0.231413025  -7.697766271  -1.434873911
  N   -1.334229587  -8.972030074   2.730208351
  H   -1.130547018  -9.089347184   3.716031936
  H   -2.179674115  -8.441933222   2.543957292
  H    1.216994210  -4.358265919  -6.948361214
  H    3.082368180  -9.437855260   1.498983399
  H    1.185061532  -9.942131080   3.067077458
  H    4.594099622  -4.738616353  -1.522864984
  H   -0.690415649  -2.352856172  -5.114759314
  H   -3.299401238  -1.857189836  -1.605112529
  H    0.123558482  -3.831357913   4.123880834
  H    1.360909659  -8.267039282  -8.458717436
  H   -1.259966719   4.296763478  -6.361237950
