%mem=64gb
%nproc=28       
%Chk=snap_189.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_189 

2     1 
  O    0.304533548  -1.655300949  -5.944950316
  C    0.493835627  -2.745190729  -6.842284880
  H    0.533689003  -3.698924997  -6.276082814
  H    1.493191125  -2.551732204  -7.286449748
  C   -0.604462671  -2.730311578  -7.913512747
  H   -0.758198080  -1.711280295  -8.334457218
  O   -0.108514802  -3.514545589  -9.050298073
  C   -0.963359984  -4.616411086  -9.324176703
  H   -0.964895707  -4.752852515 -10.432727553
  C   -1.946237327  -3.384045114  -7.488345796
  H   -1.874846326  -3.914844790  -6.510300472
  C   -2.302754147  -4.333645905  -8.641686508
  H   -3.010367062  -3.828360957  -9.343097595
  H   -2.853361007  -5.237942988  -8.316107191
  O   -2.961458680  -2.349756797  -7.471525279
  N   -0.274112967  -5.819196988  -8.705830145
  C   -0.819224010  -7.097627704  -8.472348355
  C    0.138951090  -7.804369450  -7.685730914
  N    1.244117287  -6.970026039  -7.475674925
  C    0.988732203  -5.781798237  -8.068876580
  N   -2.028645305  -7.594665120  -8.867689574
  C   -2.289723297  -8.878543320  -8.433546459
  N   -1.403599190  -9.627278828  -7.621512727
  C   -0.129249148  -9.120335557  -7.169267021
  N   -3.477952871  -9.444166611  -8.802449006
  H   -4.100439450  -8.948968238  -9.437245889
  H   -3.764955138 -10.363984425  -8.489505340
  O    0.512097190  -9.784949057  -6.382586226
  H    1.655001237  -4.909019248  -8.082053417
  H   -1.739714932 -10.492698103  -7.139275086
  P   -3.395344338  -1.840683845  -5.966385234
  O   -3.670856831  -0.231822726  -6.132700384
  C   -2.533118874   0.641747207  -6.265979214
  H   -3.030534899   1.634253446  -6.359248072
  H   -1.992599351   0.425487318  -7.209049445
  C   -1.602362127   0.577282308  -5.046764166
  H   -0.923269843  -0.326960855  -5.066669700
  O   -0.671428351   1.693791018  -5.168016667
  C   -0.507246581   2.343261464  -3.887965910
  H    0.486790034   2.006944738  -3.484598212
  C   -2.314593369   0.722076584  -3.679665217
  H   -3.427880984   0.755451912  -3.755823646
  C   -1.714193447   1.974693637  -3.023216146
  H   -1.419758482   1.741519526  -1.971235157
  H   -2.469317291   2.782734993  -2.952665644
  O   -1.883057593  -0.358735854  -2.824874562
  O   -2.209446993  -2.156012131  -5.055090642
  O   -4.847717394  -2.509030354  -5.841239166
  N   -0.425935949   3.795247270  -4.220516032
  C   -0.031309511   4.338335643  -5.438460584
  C    0.032740585   5.758016509  -5.256527178
  N   -0.321895474   6.076581754  -3.937799235
  C   -0.586178996   4.922424769  -3.327932442
  N    0.278006265   3.744676797  -6.660990557
  C    0.672962297   4.619598499  -7.689333951
  N    0.753068132   5.934629132  -7.566875750
  C    0.443941746   6.585150959  -6.351607572
  H    0.931221357   4.161883419  -8.670828968
  N    0.558604275   7.920770746  -6.296549875
  H    0.865162895   8.465189638  -7.104107641
  H    0.348216029   8.432594805  -5.438077779
  H   -0.879055607   4.788908623  -2.289496326
  P   -2.745266292  -1.764607182  -2.890595907
  O   -1.442898530  -2.734910912  -2.912803627
  C   -1.662340105  -4.167892079  -2.870168404
  H   -2.553298576  -4.481470778  -3.446337052
  H   -0.748922027  -4.538874945  -3.395971343
  C   -1.748451396  -4.611429086  -1.406239433
  H   -2.033347699  -3.768339530  -0.716553225
  O   -2.860859725  -5.543245004  -1.326446142
  C   -2.417410779  -6.848652311  -0.934426102
  H   -3.195323757  -7.201044682  -0.212346909
  C   -0.497881558  -5.347047016  -0.859550022
  H    0.322196477  -5.463350535  -1.607441150
  C   -1.014955974  -6.701820552  -0.342473015
  H   -0.326998656  -7.530621846  -0.607018580
  H   -1.060590373  -6.705287921   0.769454499
  O   -0.090515790  -4.518032822   0.246599612
  O   -3.992460232  -1.706585707  -3.740523427
  O   -3.190551581  -1.983723218  -1.334126378
  N   -2.479808742  -7.727048968  -2.156321134
  C   -1.417669836  -7.776724463  -3.104178497
  N   -1.572514672  -8.701240876  -4.167189778
  C   -2.759120399  -9.467083226  -4.425559833
  C   -3.848165156  -9.252287836  -3.464866071
  C   -3.685121682  -8.429242619  -2.399036803
  O   -0.417406268  -7.081903386  -3.034672193
  H   -0.739741277  -8.849039704  -4.759304797
  O   -2.764442886 -10.177356621  -5.414676512
  C   -5.115693266  -9.985896494  -3.731408652
  H   -5.633092346  -9.583542722  -4.617223256
  H   -4.925312541 -11.054903934  -3.935361310
  H   -5.827056217  -9.947049607  -2.894477550
  H   -4.487440154  -8.265746160  -1.664122995
  P    1.489802199  -4.597611404   0.716043568
  O    1.813987498  -2.991573230   0.619840208
  C    3.062071216  -2.572945552   0.027818811
  H    3.314713139  -1.688459144   0.656821543
  H    3.864024542  -3.334898643   0.131845557
  C    2.889994310  -2.168716035  -1.440228193
  H    3.418887166  -1.203634586  -1.645917777
  O    3.623911268  -3.165933510  -2.221190415
  C    2.927198800  -3.481941068  -3.436330915
  H    3.574566458  -3.120311411  -4.273866310
  C    1.457702982  -2.116957455  -2.010267979
  H    0.686142260  -2.534918585  -1.327670780
  C    1.541096644  -2.837975662  -3.363439910
  H    0.716143898  -3.563090935  -3.499181549
  H    1.417428527  -2.107899467  -4.201422290
  O    1.149389818  -0.726041255  -2.214990412
  O    2.347775398  -5.545555717  -0.003709876
  O    1.322219778  -4.808990874   2.315628704
  N    2.842749880  -4.982870439  -3.541455435
  C    2.420726008  -5.531321062  -4.814673358
  N    2.327013212  -6.901762346  -4.960059050
  C    2.587441208  -7.732495649  -3.893565808
  C    3.067338614  -7.197621238  -2.651412316
  C    3.203560533  -5.840517402  -2.503903301
  O    2.146654810  -4.761822905  -5.735615404
  N    2.346152991  -9.063432238  -4.093186783
  H    2.603815994  -9.756635365  -3.406937871
  H    2.030716194  -9.406215145  -4.999988043
  H   -0.472921434  -1.846761246  -5.323183414
  H    3.600046899  -5.374252299  -1.578014037
  H    3.310760419  -7.852671335  -1.813387939
  H    0.195442418  -0.668206158  -2.467603474
  H   -5.338056832  -2.349025275  -4.945030819
  H   -4.139485770  -1.827208183  -1.076860636
  H    0.850337730  -4.098161044   2.838444415
  H    1.987755772  -7.170378536  -6.741511794
  H    0.212761926   2.718392611  -6.759037304
