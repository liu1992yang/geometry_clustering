%mem=64gb
%nproc=28       
%Chk=snap_168.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_168 

2     1 
  O    1.055093660  -1.652065745  -6.222602382
  C    0.966315026  -2.847625919  -6.980561487
  H    1.129767130  -3.732002935  -6.324998370
  H    1.814110140  -2.761326743  -7.697989164
  C   -0.359750874  -2.911231572  -7.756508172
  H   -0.629392309  -1.917269492  -8.177799981
  O   -0.065694311  -3.747444307  -8.931515441
  C   -0.992889756  -4.813135614  -9.050045104
  H   -1.175106968  -4.965249950 -10.140935234
  C   -1.577422577  -3.582573833  -7.062164579
  H   -1.289637583  -4.160489466  -6.154400777
  C   -2.185934421  -4.476978749  -8.152676885
  H   -2.979432078  -3.929456413  -8.708309618
  H   -2.710019528  -5.371997153  -7.745798824
  O   -2.578354409  -2.581356597  -6.757215389
  N   -0.303138615  -6.058948841  -8.523089335
  C   -0.818414315  -7.369928255  -8.632331438
  C   -0.132824477  -8.159735646  -7.657903446
  N    0.788053884  -7.336907941  -6.999376369
  C    0.662057396  -6.085359240  -7.494293248
  N   -1.767187383  -7.837818048  -9.496071371
  C   -2.104619805  -9.157272276  -9.312870616
  N   -1.528626742  -9.987250364  -8.313553605
  C   -0.519456820  -9.518524097  -7.386552621
  N   -3.060888210  -9.690493048 -10.136143101
  H   -3.468379353  -9.117111458 -10.872964724
  H   -3.332099518 -10.662634100 -10.108400066
  O   -0.143900637 -10.257685340  -6.502735580
  H    1.247569752  -5.214497201  -7.148574879
  H   -1.861406676 -10.952040912  -8.173406537
  P   -2.546630108  -2.059503066  -5.191520793
  O   -3.049883843  -0.510962392  -5.360922879
  C   -2.243000075   0.356877924  -6.185580532
  H   -2.865512005   1.279213116  -6.202896891
  H   -2.175108550  -0.041595088  -7.217589719
  C   -0.862641955   0.625045452  -5.571727624
  H   -0.064566349  -0.083990088  -5.945507674
  O   -0.413476844   1.923319672  -6.078202781
  C   -0.105007050   2.816921575  -5.004007518
  H    0.861681692   3.305318932  -5.292108372
  C   -0.854421570   0.720689488  -4.024789359
  H   -1.889047654   0.722776658  -3.593990828
  C   -0.083399016   2.015480583  -3.709581405
  H    0.966320372   1.746058493  -3.424973431
  H   -0.476093705   2.534898876  -2.818905248
  O   -0.041698036  -0.361418393  -3.533925076
  O   -1.109421902  -2.283425039  -4.721203509
  O   -3.719323314  -2.899256829  -4.477075699
  N   -1.180932307   3.868258831  -5.049296514
  C   -1.635411617   4.470317412  -6.219667880
  C   -2.558970089   5.493975122  -5.833509952
  N   -2.683227123   5.503874776  -4.437145358
  C   -1.875292593   4.546416379  -3.980031127
  N   -1.327275239   4.231660187  -7.558773549
  C   -1.950642098   5.078821426  -8.494551944
  N   -2.796018526   6.047255874  -8.183900979
  C   -3.156521122   6.315210157  -6.844160247
  H   -1.703221737   4.915149732  -9.567602541
  N   -4.021486135   7.312660424  -6.608848537
  H   -4.406535575   7.874650374  -7.369681976
  H   -4.310376766   7.549092660  -5.658374677
  H   -1.722934077   4.281130707  -2.936873645
  P   -0.893631622  -1.490302904  -2.660912382
  O    0.068137850  -2.812789076  -2.587693885
  C   -0.560546023  -4.111109161  -2.568327041
  H   -1.272066909  -4.226335666  -3.407931983
  H    0.311148198  -4.774686042  -2.759170953
  C   -1.226221233  -4.372312548  -1.210519377
  H   -1.183969431  -3.497562592  -0.514378983
  O   -2.648441757  -4.553937291  -1.468913555
  C   -3.027483438  -5.923259182  -1.257601514
  H   -4.074766736  -5.878224260  -0.874583889
  C   -0.736895341  -5.662323704  -0.493496475
  H    0.054520150  -6.221589309  -1.056770004
  C   -2.002404392  -6.515038122  -0.292575470
  H   -1.802303265  -7.594935227  -0.451798104
  H   -2.351034174  -6.428678037   0.761898170
  O   -0.290634253  -5.264217190   0.820340434
  O   -2.373150481  -1.540729912  -2.960670385
  O   -0.577071213  -0.914152816  -1.166274604
  N   -3.041627752  -6.561735662  -2.627450376
  C   -1.854777339  -7.118407013  -3.190886671
  N   -1.834831422  -7.285955844  -4.598447197
  C   -2.924123036  -6.976760317  -5.471371935
  C   -4.130488003  -6.478824385  -4.803529397
  C   -4.144237269  -6.259334044  -3.464795195
  O   -0.877186759  -7.438368922  -2.538244092
  H   -0.955015275  -7.652213811  -4.985546227
  O   -2.731202092  -7.122917636  -6.666392969
  C   -5.321101327  -6.232509448  -5.663626221
  H   -5.775064017  -5.247055769  -5.474231707
  H   -5.070216682  -6.263733663  -6.737110563
  H   -6.099345744  -6.996783867  -5.501430921
  H   -5.028802883  -5.847407044  -2.956032464
  P    1.321311105  -5.440440628   1.096640240
  O    1.872124396  -3.984405705   0.593461091
  C    3.317086260  -3.814143186   0.627225459
  H    3.467606397  -3.023723503   1.398131021
  H    3.868393737  -4.732094036   0.919635947
  C    3.768899244  -3.328859682  -0.751847734
  H    4.670422601  -2.678881686  -0.648262721
  O    4.250484675  -4.477745629  -1.515166932
  C    3.707860574  -4.491121364  -2.844611126
  H    4.592037832  -4.607339827  -3.519087200
  C    2.657698169  -2.671911577  -1.610581181
  H    1.638403726  -2.926372250  -1.219131651
  C    2.905724341  -3.203193885  -3.029881080
  H    1.956471434  -3.351757074  -3.594453900
  H    3.452702819  -2.476986101  -3.661595250
  O    2.673658156  -1.257260635  -1.505592359
  O    1.958692240  -6.609561285   0.469844824
  O    1.392886553  -5.259826451   2.708939169
  N    2.876100446  -5.752837398  -2.970454041
  C    2.188281016  -6.024646505  -4.217926622
  N    1.686753279  -7.297933220  -4.442028378
  C    1.803684048  -8.281906397  -3.482750665
  C    2.425269864  -7.994892977  -2.222117096
  C    2.972635392  -6.759573422  -2.002918964
  O    2.046269430  -5.128263281  -5.045106133
  N    1.331192406  -9.516328467  -3.821440971
  H    1.351778444 -10.286130002  -3.168157018
  H    0.908834512  -9.698812377  -4.729223588
  H    0.522269710  -1.727301517  -5.371577600
  H    3.503866358  -6.505342439  -1.063196765
  H    2.455766480  -8.741381413  -1.424968473
  H    3.465750139  -0.856813960  -1.921990345
  H   -3.689719348  -2.857453834  -3.434533705
  H    0.392016535  -0.919608912  -0.880417250
  H    0.927764279  -5.913295452   3.297718432
  H    1.368062172  -7.610289794  -6.144957254
  H   -0.713418464   3.444135655  -7.811497641
